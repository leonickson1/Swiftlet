import Foundation
import Metal

/// State binding for the Metal fast path.
///
/// The fast path keeps each DeltaNet layer's conv history and delta
/// recurrence in per-layer GPU buffers (FastLayer.hist / .state) and each
/// attention layer's KV rows in a GPU cache indexed by absolute position.
/// Those buffers belong to the model, so at any moment they hold exactly one
/// state's live recurrence: `boundState`. Stepping a different `DecodeState`
/// first captures the outgoing state's recurrent state into its CPU fields,
/// then loads the incoming state's CPU fields (conv tail, delta state, KV
/// rows) back into the GPU buffers. The bound state's KV mirror is kept
/// current by appendKVMirror on every step, so KV never needs a capture pass.
///
/// All of this is host-side memcpy between command buffers; it adds no
/// dispatches and changes no arithmetic, so the pinned baselines and parity
/// gates are unaffected. Layouts are shared with the CPU reference (conv
/// tail [row][convDim], delta state [vHead][vDim][kDim], KV
/// [pos][kvHead][headDim]), which is what lets a state move between engines.
///
/// Before this, the fast path bound a state by identity and only re-zeroed
/// the recurrent buffers at position 0, so a second live state — or any
/// state at position > 0 that had never been bound — silently continued
/// from whatever recurrence the buffers held.
extension QwenMetalModel {
    /// Makes `state` the state whose recurrence the GPU buffers hold. Same
    /// state mid-sequence: nothing. Position 0 (fresh): the recurrent
    /// buffers are zeroed; KV rows are simply overwritten from row 0 by the
    /// following steps. Otherwise: capture the previous state, then load
    /// `state`. A state that cannot be loaded is refused before anything is
    /// captured or written, so the bound state stays bound and intact.
    func bindState(_ state: QwenCPUModel.DecodeState) throws {
        if let bound = boundState, bound === state {
            if state.position == 0 { resetGPUState() }
            return
        }
        if state.position > 0 { try validateForLoad(state) }
        if let previous = boundState {
            if previous.position > 0 { captureRecurrentState(into: previous) }
            previous.boundOn = nil
        }
        boundState = state
        state.boundOn = self
        if state.position == 0 {
            resetGPUState()
        } else {
            try loadGPUState(from: state)
        }
    }

    /// Copies every DeltaNet layer's GPU conv history and delta recurrence
    /// into `state`'s CPU fields. Only meaningful for the bound state;
    /// callers guarantee no command buffer is in flight (every step ends
    /// with commitAndWait before returning).
    func captureRecurrentState(into state: QwenCPUModel.DecodeState) {
        for li in 0..<config.numHiddenLayers {
            let fl = fastLayers[li]
            if let h = fl.hist { state.convTail[li] = Self.floats(of: h) }
            if let st = fl.state { state.deltaState[li] = Self.floats(of: st) }
        }
    }

    /// Checks that `state` (position > 0) holds every field the GPU buffers
    /// need, at the size this model's geometry gives it: conv tail and delta
    /// state per DeltaNet layer, K and V rows for its position per attention
    /// layer. Touches nothing.
    func validateForLoad(_ state: QwenCPUModel.DecodeState) throws {
        let cfg = config
        let tailFloats = (cfg.linearConvKernelDim - 1) * cfg.convDim
        let stateFloats = cfg.linearNumValueHeads * cfg.linearValueHeadDim * cfg.linearKeyHeadDim
        let kvFloats = state.position * kvRowFloats
        for li in 0..<cfg.numHiddenLayers {
            if cfg.isLinearLayer(li) {
                guard let tail = state.convTail[li], tail.count == tailFloats else {
                    throw RuntimeError.inconsistentState(
                        "layer \(li): conv tail missing or not \(tailFloats) floats")
                }
                guard let st = state.deltaState[li], st.count == stateFloats else {
                    throw RuntimeError.inconsistentState(
                        "layer \(li): delta state missing or not \(stateFloats) floats")
                }
            } else {
                guard let rows = state.kv[li], rows.k.count == kvFloats, rows.v.count == kvFloats else {
                    throw RuntimeError.inconsistentState(
                        "layer \(li): KV rows missing or not \(kvFloats) floats for position \(state.position)")
                }
            }
        }
    }

    /// Loads `state`'s CPU fields into the GPU buffers: conv tail and delta
    /// state per DeltaNet layer, KV rows per attention layer. Validates the
    /// whole state against the model geometry before touching any buffer,
    /// so a malformed state leaves the GPU state as it was.
    func loadGPUState(from state: QwenCPUModel.DecodeState) throws {
        try validateForLoad(state)
        let cfg = config
        for li in 0..<cfg.numHiddenLayers {
            if cfg.isLinearLayer(li) {
                Self.fill(fastLayers[li].hist!, with: state.convTail[li]!)
                Self.fill(fastLayers[li].state!, with: state.deltaState[li]!)
            } else {
                try ensureKVCapacity(li, rows: state.position)
                let rows = state.kv[li]!
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
                     "state field does not fit its GPU buffer")
        guard bytes > 0 else { return }
        values.withUnsafeBufferPointer {
            buffer.contents().copyMemory(from: $0.baseAddress!, byteCount: bytes)
        }
    }
}
