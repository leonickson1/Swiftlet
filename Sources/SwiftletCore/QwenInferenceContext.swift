import Foundation

/// Qwen's model-owned inference context: the per-session state both Qwen
/// engines carry across incremental steps. Grows KV rows for the GQA layers;
/// keeps a fixed-size conv tail and delta recurrence for the DeltaNet layers.
///
/// Ownership: a context belongs to the model instance that created it
/// (`makeContext`), and that instance is the only one allowed to step it.
/// On the Metal fast path the live recurrent state of the currently bound
/// context sits in the model's GPU buffers; the CPU fields here hold the
/// last captured copy, refreshed whenever the model unbinds or snapshots the
/// context (see QwenMetalModel.bindContext).
public final class QwenInferenceContext: InferenceContext {
    /// Tokens already processed (RoPE offset for the next step).
    public internal(set) var position = 0
    /// layerIndex -> appended K/V rows, laid out [pos][kvHead][headDim].
    var kv: [Int: (k: [Float], v: [Float])] = [:]
    /// layerIndex -> last (convKernel-1) rows of mixed qkv, [row][convDim].
    var convTail: [Int: [Float]] = [:]
    /// layerIndex -> delta recurrence state, [vHead][vDim][kDim].
    var deltaState: [Int: [Float]] = [:]

    /// The model instance that created this context. Weak so a context can
    /// outlive its model without keeping it alive; a context whose owner is
    /// gone is refused by every model (its live GPU state died with the
    /// owner, so continuing it anywhere else would be silently wrong).
    private weak var owner: AnyObject?

    init(owner: AnyObject) {
        self.owner = owner
    }

    public func reset() {
        position = 0
        kv.removeAll()
        convTail.removeAll()
        deltaState.removeAll()
    }

    /// Refuses use by anything but the creating model instance.
    func checkOwner(_ model: AnyObject) throws {
        guard let owner, owner === model else { throw InferenceContextError.foreignContext }
    }
}

extension QwenCPUModel {
    /// Internal spelling kept for the engines' private signatures, so the
    /// Metal file's per-layer helpers did not have to churn for the S7a
    /// seam. Public callers use `QwenInferenceContext`.
    typealias DecodeState = QwenInferenceContext
}

extension QwenCPUModel: InferenceModel {
    public var modelDir: URL { ckpt.dir }
    public var vocabSize: Int { config.vocabSize }

    public func makeContext() -> any InferenceContext { makeQwenContext() }

    /// Typed factory for callers that hold the concrete model.
    public func makeQwenContext() -> QwenInferenceContext {
        QwenInferenceContext(owner: self)
    }

    public func step(_ tokens: [Int], context: any InferenceContext) throws -> [Float] {
        guard let ctx = context as? QwenInferenceContext else {
            throw InferenceContextError.foreignContext
        }
        return try step(tokens, context: ctx)
    }
}

extension QwenMetalModel: InferenceModel {
    public var modelDir: URL { ckpt.dir }
    public var vocabSize: Int { config.vocabSize }

    public func makeContext() -> any InferenceContext { makeQwenContext() }

    /// Typed factory for callers that hold the concrete model.
    public func makeQwenContext() -> QwenInferenceContext {
        QwenInferenceContext(owner: self)
    }

    public func step(_ tokens: [Int], context: any InferenceContext) throws -> [Float] {
        guard let ctx = context as? QwenInferenceContext else {
            throw InferenceContextError.foreignContext
        }
        return try step(tokens, context: ctx)
    }
}

// MARK: - Persistence (S6)

extension QwenInferenceContext {
    /// The context's fields as snapshot sections, one per (layer, kind),
    /// in layer order. Callers on the Metal fast path capture the bound
    /// context's GPU recurrence first (the CPU fields are otherwise stale).
    func snapshotSections() -> [ConversationState.Section] {
        var sections: [ConversationState.Section] = []
        for layer in Set(kv.keys).union(convTail.keys).union(deltaState.keys).sorted() {
            if let rows = kv[layer] {
                sections.append(.init(layer: layer, kind: .k, values: rows.k))
                sections.append(.init(layer: layer, kind: .v, values: rows.v))
            }
            if let tail = convTail[layer] {
                sections.append(.init(layer: layer, kind: .convTail, values: tail))
            }
            if let state = deltaState[layer] {
                sections.append(.init(layer: layer, kind: .deltaState, values: state))
            }
        }
        return sections
    }

    /// Fills a fresh context from a decoded, verified snapshot.
    func apply(_ snapshot: ConversationState.Snapshot) {
        reset()
        for section in snapshot.sections {
            switch section.kind {
            case .k: kv[section.layer, default: (k: [], v: [])].k = section.values
            case .v: kv[section.layer, default: (k: [], v: [])].v = section.values
            case .convTail: convTail[section.layer] = section.values
            case .deltaState: deltaState[section.layer] = section.values
            }
        }
        position = snapshot.position
    }
}

extension QwenCPUModel: PersistableInferenceModel {
    /// Model identity for snapshots: config plus hashes.json or the shard
    /// headers, never weight bytes, so it is cheap enough to recompute.
    var stateIdentity: ModelIdentity {
        get throws { try ModelIdentity.compute(modelDir: ckpt.dir, shards: ckpt.shardURLs) }
    }

    public func snapshot(of context: any InferenceContext) throws -> Data {
        guard let ctx = context as? QwenInferenceContext else {
            throw InferenceContextError.foreignContext
        }
        try ctx.checkOwner(self)
        let identity = try stateIdentity
        return ConversationState.encode(.init(
            identityScheme: identity.scheme, identity: identity.digest,
            geometry: .init(config: config), position: ctx.position,
            sections: ctx.snapshotSections()))
    }

    public func restoreContext(from data: Data) throws -> any InferenceContext {
        try restoreQwenContext(from: data)
    }

    /// Typed restore for callers that hold the concrete model.
    public func restoreQwenContext(from data: Data) throws -> QwenInferenceContext {
        let snapshot = try ConversationState.decode(data)
        try ConversationState.verify(snapshot, identity: try stateIdentity, geometry: .init(config: config))
        let ctx = makeQwenContext()
        ctx.apply(snapshot)
        return ctx
    }
}

extension QwenMetalModel: PersistableInferenceModel {
    /// Model identity for snapshots: config plus hashes.json or the shard
    /// headers, never weight bytes, so it is cheap enough to recompute.
    var stateIdentity: ModelIdentity {
        get throws { try ModelIdentity.compute(modelDir: ckpt.dir, shards: ckpt.shardURLs) }
    }

    /// Serializes `context`. If it is the bound context its conv history
    /// and delta recurrence are read back from the GPU buffers first; the
    /// KV sections come from the context's mirror, which appendKVMirror
    /// copies from the GPU rows after every attention layer (see
    /// ConversationStateTests.metalSnapshotMatchesDirectGPUReadback for
    /// the bitwise proof of both). The context stays bound and live.
    public func snapshot(of context: any InferenceContext) throws -> Data {
        guard let ctx = context as? QwenInferenceContext else {
            throw InferenceContextError.foreignContext
        }
        try ctx.checkOwner(self)
        if let bound = boundContext, bound === ctx, ctx.position > 0 {
            captureRecurrentState(into: ctx)
        }
        let identity = try stateIdentity
        return ConversationState.encode(.init(
            identityScheme: identity.scheme, identity: identity.digest,
            geometry: .init(config: config), position: ctx.position,
            sections: ctx.snapshotSections()))
    }

    public func restoreContext(from data: Data) throws -> any InferenceContext {
        try restoreQwenContext(from: data)
    }

    /// Typed restore for callers that hold the concrete model. The restored
    /// context is unbound; its first step loads it into the GPU buffers.
    public func restoreQwenContext(from data: Data) throws -> QwenInferenceContext {
        let snapshot = try ConversationState.decode(data)
        try ConversationState.verify(snapshot, identity: try stateIdentity, geometry: .init(config: config))
        let ctx = makeQwenContext()
        ctx.apply(snapshot)
        return ctx
    }
}
