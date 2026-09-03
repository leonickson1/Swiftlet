import Foundation

/// A model that can decode incrementally: CPU reference or Metal runtime.
public protocol InferenceModel: AnyObject {
    var config: QwenConfig { get }
    var modelDir: URL { get }
    /// Effective capacity for one decode state. Fixed-size KV
    /// implementations override this with the smaller of trained and actually
    /// allocated capacity; dynamically growing runtimes use the trained cap.
    var contextCapacity: Int { get }
    func step(_ tokens: [Int], state: QwenCPUModel.DecodeState) throws -> [Float]
    func step(
        _ tokens: [Int],
        state: QwenCPUModel.DecodeState,
        shouldCancel: () -> Bool
    ) throws -> [Float]
}

public extension InferenceModel {
    /// CPU/reference fallback: the batched step is atomic from the caller's
    /// perspective, so cancellation is checked immediately before and after.
    /// Metal supplies a finer-grained implementation between command buffers.
    func step(
        _ tokens: [Int],
        state: QwenCPUModel.DecodeState,
        shouldCancel: () -> Bool
    ) throws -> [Float] {
        try checkGenerationCancellation(shouldCancel)
        let logits = try step(tokens, state: state)
        try checkGenerationCancellation(shouldCancel)
        return logits
    }
}

public extension InferenceModel {
    var contextCapacity: Int { config.maxPositionEmbeddings }
}

extension QwenCPUModel: InferenceModel {
    public var modelDir: URL { ckpt.dir }
}

extension QwenMetalModel: InferenceModel {
    public var modelDir: URL { ckpt.dir }
}

/// Engine-agnostic greedy generation loop shared by the CLI and server.
public final class TextGenerator: @unchecked Sendable {
    public enum Error: Swift.Error, Equatable, Sendable {
        /// `generate` was called again from a synchronous callback made by an
        /// active generation on this same generator.
        case reentrantGeneration
    }

    public let model: any InferenceModel
    public let eosTokens: Set<Int>
    /// Inference models own mutable decode/GPU scratch state. Keep every
    /// public generation entry point single-flight even when callers create
    /// several token streams from the same generator concurrently. A recursive
    /// call can reacquire this lock only long enough to reject the call instead
    /// of deadlocking inside a public token callback.
    private let generationLock = NSRecursiveLock()
    private var generationInProgress = false

    public init(model: any InferenceModel) {
        self.model = model
        eosTokens = Self.eosTokenIds(modelDir: model.modelDir)
    }

    /// EOS ids from generation_config.json / config.json (int or int array).
    public static func eosTokenIds(modelDir: URL) -> Set<Int> {
        var eos = Set<Int>()
        for name in ["generation_config.json", "config.json"] {
            guard let data = FileManager.default.contents(atPath: modelDir.appendingPathComponent(name).path),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
            if let v = obj["eos_token_id"] as? Int { eos.insert(v) }
            if let vs = obj["eos_token_id"] as? [Int] { eos.formUnion(vs) }
        }
        return eos
    }

    public struct Stats: Sendable {
        public var promptTokens = 0
        public var generatedTokens = 0
        public var prefillSeconds = 0.0
        public var decodeSeconds = 0.0
    }

    /// Token-id stream over `generate`, in the `AsyncStream` shape iOS chat
    /// UIs consume (Priv AI's engines all expose `-> AsyncStream`). Runs the
    /// blocking decode loop on a utility queue; cancelling the consuming task
    /// stops prefill/decode at the next model-safe boundary.
    public func tokenStream(promptIds: [Int], maxNew: Int) -> AsyncThrowingStream<Int, Swift.Error> {
        tokenStream(
            promptIds: promptIds,
            maxNew: maxNew,
            cancellation: GenerationCancellation()
        )
    }

    /// Token stream with an explicit cancellation owner for embedding clients
    /// that need to cancel independently of the consuming Task.
    public func tokenStream(
        promptIds: [Int],
        maxNew: Int,
        cancellation: GenerationCancellation
    ) -> AsyncThrowingStream<Int, Swift.Error> {
        AsyncThrowingStream { continuation in
            let work = DispatchWorkItem {
                do {
                    try self.generate(
                        promptIds: promptIds,
                        maxNew: maxNew,
                        cancellation: cancellation
                    ) { token in
                        if case .terminated = continuation.yield(token) {
                            cancellation.cancel()
                            return false
                        }
                        return true
                    }
                    continuation.finish()
                } catch GenerationInterruption.cancelled {
                    // Task/owner cancellation is normal stream termination;
                    // the local DecodeState dies with this work item.
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { termination in
                if case .cancelled = termination { cancellation.cancel() }
            }
            DispatchQueue.global(qos: .userInitiated).async(execute: work)
        }
    }

    /// Greedy-decode up to `maxNew` tokens after `promptIds`. Calls `onToken`
    /// for each generated token; return false from it to stop early. EOS is
    /// consumed as a stop signal and not reported.
    @discardableResult
    public func generate(promptIds: [Int], maxNew: Int, onToken: (Int) -> Bool) throws -> Stats {
        try generate(
            promptIds: promptIds,
            maxNew: maxNew,
            cancellation: GenerationCancellation(),
            onToken: onToken
        )
    }

    /// Cancellable form of `generate`. Any cancellation error invalidates the
    /// method-local DecodeState, which is discarded while unwinding. Concurrent
    /// callers are serialized; a call made recursively from `onToken` throws
    /// `Error.reentrantGeneration`.
    @discardableResult
    public func generate(
        promptIds: [Int],
        maxNew: Int,
        cancellation: GenerationCancellation,
        onToken: (Int) -> Bool
    ) throws -> Stats {
        generationLock.lock()
        if generationInProgress {
            generationLock.unlock()
            throw Error.reentrantGeneration
        }
        generationInProgress = true
        defer {
            generationInProgress = false
            generationLock.unlock()
        }

        // A stream can be cancelled while it waits behind another generation.
        // Observe that before allocating or passing any DecodeState to the
        // shared model.
        try checkGenerationCancellation { cancellation.isCancelled }

        let admittedMaxNew = try ContextWindow(maximumTokens: model.contextCapacity)
            .admittedMaxNew(
                processedTokens: 0,
                incomingTokens: promptIds.count,
                requestedMaxNew: maxNew
            )
        var stats = Stats()
        stats.promptTokens = promptIds.count
        let state = QwenCPUModel.DecodeState()

        let prefillStart = Date()
        var logits = try model.step(
            promptIds,
            state: state,
            shouldCancel: { cancellation.isCancelled }
        )
        stats.prefillSeconds = -prefillStart.timeIntervalSinceNow

        let decodeStart = Date()
        for _ in 0..<admittedMaxNew {
            try checkGenerationCancellation { cancellation.isCancelled }
            var best = 0
            for v in 1..<model.config.vocabSize where logits[v] > logits[best] { best = v }
            if eosTokens.contains(best) { break }
            stats.generatedTokens += 1
            if !onToken(best) { break }
            logits = try model.step(
                [best],
                state: state,
                shouldCancel: { cancellation.isCancelled }
            )
        }
        stats.decodeSeconds = -decodeStart.timeIntervalSinceNow
        return stats
    }
}
