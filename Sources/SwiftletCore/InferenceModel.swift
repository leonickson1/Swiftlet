import Foundation

/// Opaque, model-owned inference state: whatever a backend needs to continue
/// a sequence (attention KV, recurrent state, position...). Callers hold the
/// handle and pass it back to the model that created it; they never look
/// inside. This is the S7 seam: a new backend supplies its own context type
/// and nothing above the model protocol has to know what it holds.
public protocol InferenceContext: AnyObject {
    /// Tokens the context has consumed so far.
    var position: Int { get }
    /// Discards everything the context holds; the next step starts a fresh
    /// sequence. The handle stays valid.
    func reset()
}

/// Thrown when a context is handed to a model that did not create it (a
/// different backend, or a different instance of the same backend), or when
/// a context's contents cannot belong to the model's geometry.
public enum InferenceContextError: Swift.Error, Equatable {
    /// The context was created by another model instance or backend.
    case foreignContext
    /// The context's stored state does not match the model's geometry
    /// (e.g. a KV layer with the wrong row count for its position).
    case inconsistentContext(String)
}

/// A model that can decode incrementally: CPU reference or Metal runtime.
/// Contexts are created by the model, advanced by `step`, reset through
/// the context itself, and refused by any other model instance.
public protocol InferenceModel: AnyObject {
    /// Number of logits `step` returns.
    var vocabSize: Int { get }
    var modelDir: URL { get }
    /// A fresh context at position 0, owned by this model instance.
    func makeContext() -> any InferenceContext
    /// Processes `tokens` continuing from `context`, returning the last
    /// position's logits. Used for prefill (many tokens) and decode (one).
    func step(_ tokens: [Int], context: any InferenceContext) throws -> [Float]
}

/// A model whose contexts can be written out and reopened later (S6). The
/// bytes are a versioned snapshot that names the model they came from; a
/// restore into a different model, format version, geometry, or dtype is
/// refused rather than continued from foreign state.
public protocol PersistableInferenceModel: InferenceModel {
    /// Serializes `context`, which must belong to this model instance. The
    /// context stays live and unchanged.
    func snapshot(of context: any InferenceContext) throws -> Data
    /// A new context of this model holding the snapshot's state, ready to
    /// continue from its position.
    func restoreContext(from data: Data) throws -> any InferenceContext
}
