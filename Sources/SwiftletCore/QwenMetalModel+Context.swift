import Foundation
import Metal

/// Context binding for the Metal fast path.
///
/// The fast path keeps each DeltaNet layer's conv history and delta
/// recurrence in per-layer GPU buffers (FastLayer.hist / .state) and each
/// attention layer's KV rows in a GPU cache indexed by absolute position.
/// Those buffers belong to the model, so at any moment they hold exactly one
/// context's live state: `boundContext`. Stepping a different context first
/// captures the outgoing context's recurrent state into its CPU fields, then
/// loads the incoming context's CPU fields (conv tail, delta state, KV rows)
/// back into the GPU buffers. The bound context's KV mirror is kept current
/// by appendKVMirror on every step, so KV never needs a capture pass.
///
/// All of this is host-side memcpy between command buffers; it adds no
/// dispatches and changes no arithmetic, so the pinned baselines and parity
/// gates are unaffected. Layouts are shared with the CPU reference (conv
/// tail [row][convDim], delta state [vHead][vDim][kDim], KV
/// [pos][kvHead][headDim]), which is what lets a context move between
/// engines through a snapshot.
extension QwenMetalModel {
    /// Makes `ctx` the context whose state the GPU buffers hold. Same
    /// context mid-sequence: nothing. Position 0 (fresh or reset): the
    /// recurrent buffers are zeroed; KV rows are simply overwritten from
    /// row 0 by the following steps. Otherwise: capture the previous
    /// context, then load `ctx`.
    func bindContext(_ ctx: QwenInferenceContext) throws {
        if let bound = boundContext, bound === ctx {
            if ctx.position == 0 { resetGPUState() }
            return
        }
        if let previous = boundContext, previous.position > 0 {
            captureRecurrentState(into: previous)
        }
        boundContext = ctx
        if ctx.position == 0 {
            resetGPUState()
        } else {
            try loadGPUState(from: ctx)
        }
    }

    /// Copies every DeltaNet layer's GPU conv history and delta recurrence
    /// into `ctx`'s CPU fields. Only meaningful for the bound context;
    /// callers guarantee no command buffer is in flight.
    func captureRecurrentState(into ctx: QwenInferenceContext) {
        for li in 0..<config.numHiddenLayers {
            let fl = fastLayers[li]
            if let h = fl.hist { ctx.convTail[li] = Self.floats(of: h) }
            if let st = fl.state { ctx.deltaState[li] = Self.floats(of: st) }
        }
    }

    /// Loads `ctx`'s CPU fields into the GPU buffers: conv tail and delta
    /// state per DeltaNet layer, KV rows per attention layer. Validates the
    /// whole context against the model geometry before touching any buffer,
    /// so a malformed context leaves the GPU state as it was.
    func loadGPUState(from ctx: QwenInferenceContext) throws {
        let cfg = config
        let tailFloats = (cfg.linearConvKernelDim - 1) * cfg.convDim
        let stateFloats = cfg.linearNumValueHeads * cfg.linearValueHeadDim * cfg.linearKeyHeadDim
        let kvFloats = ctx.position * kvRowFloats
        for li in 0..<cfg.numHiddenLayers {
            if cfg.isLinearLayer(li) {
                guard let tail = ctx.convTail[li], tail.count == tailFloats else {
                    throw InferenceContextError.inconsistentContext(
                        "layer \(li): conv tail missing or not \(tailFloats) floats")
                }
                guard let st = ctx.deltaState[li], st.count == stateFloats else {
                    throw InferenceContextError.inconsistentContext(
                        "layer \(li): delta state missing or not \(stateFloats) floats")
                }
            } else {
                guard let rows = ctx.kv[li], rows.k.count == kvFloats, rows.v.count == kvFloats else {
                    throw InferenceContextError.inconsistentContext(
                        "layer \(li): KV rows missing or not \(kvFloats) floats for position \(ctx.position)")
                }
            }
        }
        for li in 0..<cfg.numHiddenLayers {
            if cfg.isLinearLayer(li) {
                Self.fill(fastLayers[li].hist!, with: ctx.convTail[li]!)
                Self.fill(fastLayers[li].state!, with: ctx.deltaState[li]!)
            } else {
                try ensureKVCapacity(li, rows: ctx.position)
                let rows = ctx.kv[li]!
                Self.fill(fastLayers[li].kCache!, with: rows.k, prefixOnly: true)
                Self.fill(fastLayers[li].vCache!, with: rows.v, prefixOnly: true)
            }
        }
    }

    private static func floats(of buffer: MTLBuffer) -> [Float] {
        let n = buffer.length / MemoryLayout<Float>.stride
        return Array(UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: Float.self, capacity: n), count: n))
    }

    /// Writes `values` at the start of `buffer`. Whole-buffer by default;
    /// `prefixOnly` for the KV caches, whose capacity exceeds the rows held.
    private static func fill(_ buffer: MTLBuffer, with values: [Float], prefixOnly: Bool = false) {
        let bytes = values.count * MemoryLayout<Float>.stride
        precondition(prefixOnly ? bytes <= buffer.length : bytes == buffer.length,
                     "context field does not fit its GPU buffer")
        guard bytes > 0 else { return }
        values.withUnsafeBufferPointer {
            buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: bytes)
        }
    }
}
