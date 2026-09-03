import Accelerate
import Foundation

/// Float32 CPU reference implementation of the Qwen3-Next / Qwen3.5-MoE hybrid
/// forward pass. Mirrors mlx-lm's `qwen3_next.py` + `gated_delta.py` (vendored
/// in `references/`) op for op. This is the correctness oracle for the Metal
/// runtime; it favors clarity over speed.
///
/// Weights load lazily through `Checkpoint` (plain or MLX-quantized, single or
/// sharded), one layer at a time, with routed experts dequantized on demand as
/// a batch union. That keeps the real 80B validatable on a 24 GB machine.
///
/// Layout note: expects mlx-lm runtime-format weights (as produced by
/// `mlx_lm convert` or `scripts/gen_fixtures.py`): norm weights already
/// non-centered, experts stacked into `switch_mlp.*`, conv1d as (C, K, 1).
public final class QwenCPUModel {

    // MARK: - Ops

    static func linear(_ x: [Float], rows: Int, inDim: Int, weight: [Float], outDim: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows * outDim)
        // y = x @ W^T with W stored row-major (outDim, inDim).
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            Int32(rows), Int32(outDim), Int32(inDim),
            1, x, Int32(inDim), weight, Int32(inDim),
            0, &out, Int32(outDim)
        )
        return out
    }

    static func matvec(_ weight: [Float], _ weightOffset: Int, outDim: Int, inDim: Int, _ x: [Float], _ xOffset: Int, into out: inout [Float], _ outOffset: Int) {
        weight.withUnsafeBufferPointer { w in
            x.withUnsafeBufferPointer { xp in
                out.withUnsafeMutableBufferPointer { o in
                    cblas_sgemv(
                        CblasRowMajor, CblasNoTrans, Int32(outDim), Int32(inDim),
                        1, w.baseAddress! + weightOffset, Int32(inDim),
                        xp.baseAddress! + xOffset, 1,
                        0, o.baseAddress! + outOffset, 1
                    )
                }
            }
        }
    }

    static func rmsNorm(_ x: inout [Float], rows: Int, dim: Int, weight: [Float]?, eps: Float) {
        for r in 0..<rows {
            let base = r * dim
            var ss: Float = 0
            for i in 0..<dim { ss += x[base + i] * x[base + i] }
            let scale = 1 / (ss / Float(dim) + eps).squareRoot()
            if let w = weight {
                for i in 0..<dim { x[base + i] *= scale * w[i] }
            } else {
                for i in 0..<dim { x[base + i] *= scale }
            }
        }
    }

    @inline(__always) static func sigmoid(_ v: Float) -> Float { 1 / (1 + expf(-v)) }
    @inline(__always) static func silu(_ v: Float) -> Float { v * sigmoid(v) }
    @inline(__always) static func softplus(_ v: Float) -> Float { v > 20 ? v : log1pf(expf(v)) }

    static func softmaxRow(_ x: inout [Float], base: Int, count: Int) {
        var m = -Float.infinity
        for i in 0..<count { m = max(m, x[base + i]) }
        var sum: Float = 0
        for i in 0..<count { x[base + i] = expf(x[base + i] - m); sum += x[base + i] }
        for i in 0..<count { x[base + i] /= sum }
    }

    // MARK: - Lazily loaded per-layer weights

    struct AttnWeights {
        var qProj, kProj, vProj, oProj: [Float]
        var qNorm, kNorm: [Float]
    }

    struct DeltaWeights {
        // fusedInterleaved layout (qwen3_next): in_proj_qkvz + in_proj_ba.
        var qkvz: [Float]?
        var ba: [Float]?
        // split layout (qwen3_5): in_proj_qkv (flat [q|k|v]) + z/b/a projections.
        var qkv: [Float]?
        var zProj: [Float]?
        var bProj: [Float]?
        var aProj: [Float]?
        var conv: [Float]           // (convDim, K, 1) flattened
        var dtBias, aLog: [Float]
        var norm: [Float]           // gated RMSNorm weight, per v-head-dim
        var outProj: [Float]
    }

    struct MoEWeights {
        var gate: [Float]                       // (E, hidden), dequantized
        var expertPrefix: String                // "...mlp.switch_mlp."
        var sharedGateProj, sharedUpProj: [Float]
        var sharedDownProj: [Float]
        var sharedExpertGate: [Float]           // (1, hidden)
    }

    struct LayerWeights {
        var inputNorm, postAttnNorm: [Float]
        var attn: AttnWeights?
        var delta: DeltaWeights?
        var moe: MoEWeights
    }

    public let config: QwenConfig
    let ckpt: Checkpoint
    var embed: [Float]
    var finalNorm: [Float]
    var lmHead: [Float]
    private var layerCache: (index: Int, weights: LayerWeights)?
    private var allLayers: [Int: LayerWeights] = [:]
    /// Keep every layer's dense weights resident (f32). Costs ~4 bytes/dense
    /// param but avoids re-dequantizing per decode step. ~10 GB for the 80B.
    public var retainAllLayers = false

    public init(modelDir: URL) throws {
        config = try QwenConfig(url: modelDir.appendingPathComponent("config.json"))
        ckpt = try Checkpoint(dir: modelDir)
        embed = try ckpt.moduleWeight("model.embed_tokens")
        finalNorm = try ckpt.tensor("model.norm.weight")
        lmHead = config.tieWordEmbeddings ? embed : try ckpt.moduleWeight("lm_head")
    }

    func layerWeights(_ i: Int) throws -> LayerWeights {
        if let retained = allLayers[i] { return retained }
        if let cached = layerCache, cached.index == i { return cached.weights }
        let p = "model.layers.\(i)."
        let moe = MoEWeights(
            gate: try ckpt.moduleWeight(p + "mlp.gate"),
            expertPrefix: p + "mlp.switch_mlp.",
            sharedGateProj: try ckpt.moduleWeight(p + "mlp.shared_expert.gate_proj"),
            sharedUpProj: try ckpt.moduleWeight(p + "mlp.shared_expert.up_proj"),
            sharedDownProj: try ckpt.moduleWeight(p + "mlp.shared_expert.down_proj"),
            sharedExpertGate: try ckpt.moduleWeight(p + "mlp.shared_expert_gate")
        )
        var layer = LayerWeights(
            inputNorm: try ckpt.tensor(p + "input_layernorm.weight"),
            postAttnNorm: try ckpt.tensor(p + "post_attention_layernorm.weight"),
            moe: moe
        )
        if config.isLinearLayer(i) {
            var delta = DeltaWeights(
                conv: try ckpt.tensor(p + "linear_attn.conv1d.weight"),
                dtBias: try ckpt.tensor(p + "linear_attn.dt_bias"),
                aLog: try ckpt.tensor(p + "linear_attn.A_log"),
                norm: try ckpt.tensor(p + "linear_attn.norm.weight"),
                outProj: try ckpt.moduleWeight(p + "linear_attn.out_proj")
            )
            switch config.deltaLayout {
            case .fusedInterleaved:
                delta.qkvz = try ckpt.moduleWeight(p + "linear_attn.in_proj_qkvz")
                delta.ba = try ckpt.moduleWeight(p + "linear_attn.in_proj_ba")
            case .split:
                delta.qkv = try ckpt.moduleWeight(p + "linear_attn.in_proj_qkv")
                delta.zProj = try ckpt.moduleWeight(p + "linear_attn.in_proj_z")
                delta.bProj = try ckpt.moduleWeight(p + "linear_attn.in_proj_b")
                delta.aProj = try ckpt.moduleWeight(p + "linear_attn.in_proj_a")
            }
            layer.delta = delta
        } else {
            layer.attn = AttnWeights(
                qProj: try ckpt.moduleWeight(p + "self_attn.q_proj"),
                kProj: try ckpt.moduleWeight(p + "self_attn.k_proj"),
                vProj: try ckpt.moduleWeight(p + "self_attn.v_proj"),
                oProj: try ckpt.moduleWeight(p + "self_attn.o_proj"),
                qNorm: try ckpt.tensor(p + "self_attn.q_norm.weight"),
                kNorm: try ckpt.tensor(p + "self_attn.k_norm.weight")
            )
        }
        if retainAllLayers { allLayers[i] = layer } else { layerCache = (i, layer) }
        return layer
    }

    // MARK: - Decode state (carried across incremental steps)

    /// Per-session recurrent state: grows KV for the GQA layers, fixed-size
    /// conv tail + delta state for the DeltaNet layers.
    public final class DecodeState {
        /// Tokens already processed (RoPE offset for the next step).
        public internal(set) var position = 0
        /// layerIndex -> appended K/V rows, laid out [pos][kvHead][headDim].
        var kv: [Int: (k: [Float], v: [Float])] = [:]
        /// layerIndex -> last (convKernel-1) rows of mixed qkv, [row][convDim].
        var convTail: [Int: [Float]] = [:]
        /// layerIndex -> delta recurrence state, [vHead][vDim][kDim].
        var deltaState: [Int: [Float]] = [:]
        public init() {}
    }

    // MARK: - Forward

    public struct Captures {
        public var embed: [Float] = []
        public var layerOutputs: [[Float]] = []
        public var finalNorm: [Float] = []
        public var logits: [Float] = []
        public var greedy: [Int] = []
    }

    public func forward(tokens: [Int], captureLayers: Bool = true) throws -> Captures {
        let cfg = config
        let S = tokens.count
        let D = cfg.hiddenSize
        var caps = Captures()
        let state = DecodeState()

        var h = [Float](repeating: 0, count: S * D)
        for (s, t) in tokens.enumerated() {
            for i in 0..<D { h[s * D + i] = embed[t * D + i] }
        }
        caps.embed = h

        for li in 0..<cfg.numHiddenLayers {
            h = try layerForward(h, S: S, layerIndex: li, state: state)
            if captureLayers { caps.layerOutputs.append(h) }
        }

        Self.rmsNorm(&h, rows: S, dim: D, weight: finalNorm, eps: Float(cfg.rmsNormEps))
        caps.finalNorm = h
        let logits = Self.linear(h, rows: S, inDim: D, weight: lmHead, outDim: cfg.vocabSize)
        caps.logits = logits
        for s in 0..<S {
            var best = 0
            for v in 1..<cfg.vocabSize where logits[s * cfg.vocabSize + v] > logits[s * cfg.vocabSize + best] { best = v }
            caps.greedy.append(best)
        }
        return caps
    }

    /// Incremental step: process `tokens` continuing from `state`, returning
    /// the last position's logits. Used for prefill (many tokens) and decode
    /// (one token at a time).
    public func step(_ tokens: [Int], state: DecodeState) throws -> [Float] {
        try step(tokens, state: state, shouldCancel: { false })
    }

    /// Cancellable incremental step. The reference path checks only between
    /// complete decoder layers, where no BLAS operation is in flight. A
    /// cancellation after an earlier layer may leave `state` partially
    /// updated, so the caller must discard it (SwiftletSession does).
    public func step(
        _ tokens: [Int],
        state: DecodeState,
        shouldCancel: () -> Bool
    ) throws -> [Float] {
        try checkGenerationCancellation(shouldCancel)
        let cfg = config
        let S = tokens.count
        let D = cfg.hiddenSize
        try ContextWindow(maximumTokens: cfg.maxPositionEmbeddings).validateStep(
            processedTokens: state.position, incomingTokens: S
        )

        var h = [Float](repeating: 0, count: S * D)
        for (s, t) in tokens.enumerated() {
            for i in 0..<D { h[s * D + i] = embed[t * D + i] }
        }
        for li in 0..<cfg.numHiddenLayers {
            try checkGenerationCancellation(shouldCancel)
            h = try layerForward(h, S: S, layerIndex: li, state: state)
        }
        try checkGenerationCancellation(shouldCancel)
        state.position += S

        var last = Array(h[(S - 1) * D..<S * D])
        Self.rmsNorm(&last, rows: 1, dim: D, weight: finalNorm, eps: Float(cfg.rmsNormEps))
        try checkGenerationCancellation(shouldCancel)
        let logits = Self.linear(last, rows: 1, inDim: D, weight: lmHead, outDim: cfg.vocabSize)
        try checkGenerationCancellation(shouldCancel)
        return logits
    }

    /// One decoder layer: norm -> (DeltaNet | gated GQA) -> residual -> norm -> MoE -> residual.
    /// Passing a fresh `DecodeState` reproduces the stateless whole-sequence pass.
    public func layerForward(_ h: [Float], S: Int, layerIndex: Int, state: DecodeState? = nil) throws -> [Float] {
        let cfg = config
        let D = cfg.hiddenSize
        let layer = try layerWeights(layerIndex)
        let state = state ?? DecodeState()
        var h = h

        var x = h
        Self.rmsNorm(&x, rows: S, dim: D, weight: layer.inputNorm, eps: Float(cfg.rmsNormEps))
        let r: [Float]
        if let delta = layer.delta {
            r = deltaNetForward(x, S: S, w: delta, layerIndex: layerIndex, state: state)
        } else {
            r = attentionForward(x, S: S, w: layer.attn!, layerIndex: layerIndex, state: state)
        }
        for i in 0..<h.count { h[i] += r[i] }

        var x2 = h
        Self.rmsNorm(&x2, rows: S, dim: D, weight: layer.postAttnNorm, eps: Float(cfg.rmsNormEps))
        let m = try moeForward(x2, S: S, w: layer.moe)
        for i in 0..<h.count { h[i] += m[i] }
        return h
    }

    // MARK: - Gated full attention (GQA layers)

    func attentionForward(_ x: [Float], S: Int, w: AttnWeights, layerIndex: Int, state: DecodeState) -> [Float] {
        let cfg = config
        let H = cfg.numAttentionHeads, KVH = cfg.numKeyValueHeads, hd = cfg.headDim
        let eps = Float(cfg.rmsNormEps)
        let past = state.position

        // q_proj emits (H, 2*hd) per position: first hd = query, second hd = output gate.
        let qOut = Self.linear(x, rows: S, inDim: cfg.hiddenSize, weight: w.qProj, outDim: H * hd * 2)
        var q = [Float](repeating: 0, count: S * H * hd)
        var gate = [Float](repeating: 0, count: S * H * hd)
        for s in 0..<S {
            for head in 0..<H {
                let src = s * H * hd * 2 + head * 2 * hd
                for i in 0..<hd {
                    q[(s * H + head) * hd + i] = qOut[src + i]
                    gate[(s * H + head) * hd + i] = qOut[src + hd + i]
                }
            }
        }
        var k = Self.linear(x, rows: S, inDim: cfg.hiddenSize, weight: w.kProj, outDim: KVH * hd)
        var v = Self.linear(x, rows: S, inDim: cfg.hiddenSize, weight: w.vProj, outDim: KVH * hd)

        Self.rmsNorm(&q, rows: S * H, dim: hd, weight: w.qNorm, eps: eps)
        Self.rmsNorm(&k, rows: S * KVH, dim: hd, weight: w.kNorm, eps: eps)

        applyRope(&q, S: S, heads: H, offset: past)
        applyRope(&k, S: S, heads: KVH, offset: past)

        // Append new rows to the layer's KV cache.
        var cache = state.kv[layerIndex] ?? (k: [], v: [])
        cache.k.append(contentsOf: k)
        cache.v.append(contentsOf: v)
        state.kv[layerIndex] = cache
        let kAll = cache.k, vAll = cache.v

        // Causal softmax attention per head over past + new; GQA maps head -> kv head.
        let scale = 1 / Float(hd).squareRoot()
        var attnOut = [Float](repeating: 0, count: S * H * hd)
        let group = H / KVH
        var scores = [Float](repeating: 0, count: past + S)
        for head in 0..<H {
            let kvHead = head / group
            for si in 0..<S {
                let kvLen = past + si + 1
                for sj in 0..<kvLen {
                    var dot: Float = 0
                    for i in 0..<hd { dot += q[(si * H + head) * hd + i] * kAll[(sj * KVH + kvHead) * hd + i] }
                    scores[sj] = dot * scale
                }
                Self.softmaxRow(&scores, base: 0, count: kvLen)
                let dst = (si * H + head) * hd
                for sj in 0..<kvLen {
                    let p = scores[sj]
                    let vBase = (sj * KVH + kvHead) * hd
                    for i in 0..<hd { attnOut[dst + i] += p * vAll[vBase + i] }
                }
            }
        }

        for i in 0..<attnOut.count { attnOut[i] *= Self.sigmoid(gate[i]) }
        return Self.linear(attnOut, rows: S, inDim: H * hd, weight: w.oProj, outDim: cfg.hiddenSize)
    }

    /// Partial NeoX-style RoPE over the first `rotaryDims` of each head.
    func applyRope(_ x: inout [Float], S: Int, heads: Int, offset: Int) {
        let hd = config.headDim
        let rot = config.rotaryDims
        let half = rot / 2
        for s in 0..<S {
            for head in 0..<heads {
                let base = (s * heads + head) * hd
                for j in 0..<half {
                    let invFreq = powf(Float(config.ropeTheta), -Float(2 * j) / Float(rot))
                    let angle = Float(offset + s) * invFreq
                    let c = cosf(angle), sn = sinf(angle)
                    let a = x[base + j]
                    let b = x[base + half + j]
                    x[base + j] = a * c - b * sn
                    x[base + half + j] = b * c + a * sn
                }
            }
        }
    }

    // MARK: - Gated DeltaNet (linear layers)

    func deltaNetForward(_ x: [Float], S: Int, w: DeltaWeights, layerIndex: Int, state: DecodeState) -> [Float] {
        let cfg = config
        let nk = cfg.linearNumKeyHeads, nv = cfg.linearNumValueHeads
        let dk = cfg.linearKeyHeadDim, dv = cfg.linearValueHeadDim
        let rep = nv / nk
        let keyDim = cfg.keyDim, valueDim = cfg.valueDim, convDim = cfg.convDim
        let K = cfg.linearConvKernelDim

        var mixedQKV: [Float]
        var z: [Float]
        var bArr: [Float]
        var aArr: [Float]

        switch cfg.deltaLayout {
        case .fusedInterleaved:
            // Interleaved per-k-head layout: [q(dk), k(dk), v(rep*dv), z(rep*dv)] per k-head.
            let qkvz = Self.linear(x, rows: S, inDim: cfg.hiddenSize, weight: w.qkvz!, outDim: 2 * keyDim + 2 * valueDim)
            let ba = Self.linear(x, rows: S, inDim: cfg.hiddenSize, weight: w.ba!, outDim: 2 * nv)
            let chunk = 2 * dk + 2 * rep * dv
            mixedQKV = [Float](repeating: 0, count: S * convDim)   // [q | k | v] head-major
            z = [Float](repeating: 0, count: S * valueDim)
            bArr = [Float](repeating: 0, count: S * nv)
            aArr = [Float](repeating: 0, count: S * nv)
            for s in 0..<S {
                for hk in 0..<nk {
                    let src = s * (2 * keyDim + 2 * valueDim) + hk * chunk
                    for i in 0..<dk {
                        mixedQKV[s * convDim + hk * dk + i] = qkvz[src + i]
                        mixedQKV[s * convDim + keyDim + hk * dk + i] = qkvz[src + dk + i]
                    }
                    for ri in 0..<rep {
                        let hv = hk * rep + ri
                        for i in 0..<dv {
                            mixedQKV[s * convDim + 2 * keyDim + hv * dv + i] = qkvz[src + 2 * dk + ri * dv + i]
                            z[s * valueDim + hv * dv + i] = qkvz[src + 2 * dk + rep * dv + ri * dv + i]
                        }
                    }
                    let baSrc = s * 2 * nv + hk * 2 * rep
                    for ri in 0..<rep {
                        bArr[s * nv + hk * rep + ri] = ba[baSrc + ri]
                        aArr[s * nv + hk * rep + ri] = ba[baSrc + rep + ri]
                    }
                }
            }
        case .split:
            // qwen3_5: in_proj_qkv already emits flat [q | k | v]; z/b/a separate.
            mixedQKV = Self.linear(x, rows: S, inDim: cfg.hiddenSize, weight: w.qkv!, outDim: convDim)
            z = Self.linear(x, rows: S, inDim: cfg.hiddenSize, weight: w.zProj!, outDim: valueDim)
            bArr = Self.linear(x, rows: S, inDim: cfg.hiddenSize, weight: w.bProj!, outDim: nv)
            aArr = Self.linear(x, rows: S, inDim: cfg.hiddenSize, weight: w.aProj!, outDim: nv)
        }

        // Causal depthwise conv1d (kernel K) + silu, continuing from the
        // session's conv tail (zeros on a fresh session).
        // mlx Conv1d weight is (C, K, 1): w.conv[c * K + j].
        let tailRows = K - 1
        var padded = state.convTail[layerIndex] ?? [Float](repeating: 0, count: tailRows * convDim)
        padded.append(contentsOf: mixedQKV)
        var convOut = [Float](repeating: 0, count: S * convDim)
        for s in 0..<S {
            for c in 0..<convDim {
                var acc: Float = 0
                for j in 0..<K {
                    acc += w.conv[c * K + j] * padded[(s + j) * convDim + c]
                }
                convOut[s * convDim + c] = Self.silu(acc)
            }
        }
        state.convTail[layerIndex] = Array(padded.suffix(tailRows * convDim))

        // Split conv output and normalize q, k per head (no weight, eps 1e-6),
        // with q scaled by 1/dk and k by 1/sqrt(dk) (mlx: inv_scale^2, inv_scale).
        var qh = [Float](repeating: 0, count: S * keyDim)
        var kh = [Float](repeating: 0, count: S * keyDim)
        var vh = [Float](repeating: 0, count: S * valueDim)
        for s in 0..<S {
            for i in 0..<keyDim {
                qh[s * keyDim + i] = convOut[s * convDim + i]
                kh[s * keyDim + i] = convOut[s * convDim + keyDim + i]
            }
            for i in 0..<valueDim { vh[s * valueDim + i] = convOut[s * convDim + 2 * keyDim + i] }
        }
        Self.rmsNorm(&qh, rows: S * nk, dim: dk, weight: nil, eps: 1e-6)
        Self.rmsNorm(&kh, rows: S * nk, dim: dk, weight: nil, eps: 1e-6)
        let invScale = 1 / Float(dk).squareRoot()
        for i in 0..<qh.count { qh[i] *= invScale * invScale }
        for i in 0..<kh.count { kh[i] *= invScale }

        // Gated delta rule recurrence (references/gated_delta.py, ops path),
        // continuing from the session's persistent state.
        var delta0 = state.deltaState[layerIndex] ?? [Float](repeating: 0, count: nv * dv * dk)
        var out = [Float](repeating: 0, count: S * valueDim)
        for s in 0..<S {
            for hv in 0..<nv {
                let hk = hv / rep
                let g = expf(-expf(w.aLog[hv]) * Self.softplus(aArr[s * nv + hv] + w.dtBias[hv]))
                let beta = Self.sigmoid(bArr[s * nv + hv])
                let qBase = s * keyDim + hk * dk
                let kBase = s * keyDim + hk * dk
                let vBase = s * valueDim + hv * dv
                for dvi in 0..<dv {
                    let stBase = (hv * dv + dvi) * dk
                    var kvMem: Float = 0
                    for dki in 0..<dk {
                        delta0[stBase + dki] *= g
                        kvMem += delta0[stBase + dki] * kh[kBase + dki]
                    }
                    let delta = (vh[vBase + dvi] - kvMem) * beta
                    var y: Float = 0
                    for dki in 0..<dk {
                        delta0[stBase + dki] += kh[kBase + dki] * delta
                        y += delta0[stBase + dki] * qh[qBase + dki]
                    }
                    out[s * valueDim + hv * dv + dvi] = y
                }
            }
        }
        state.deltaState[layerIndex] = delta0

        // Gated RMSNorm: silu(z) * (rmsnorm(out) * weight), per v-head.
        var normed = out
        Self.rmsNorm(&normed, rows: S * nv, dim: dv, weight: w.norm, eps: Float(cfg.rmsNormEps))
        for i in 0..<normed.count { normed[i] *= Self.silu(z[i]) }

        return Self.linear(normed, rows: S, inDim: valueDim, weight: w.outProj, outDim: cfg.hiddenSize)
    }

    // MARK: - Mixture of experts

    func moeForward(_ x: [Float], S: Int, w: MoEWeights) throws -> [Float] {
        let cfg = config
        let D = cfg.hiddenSize, E = cfg.numExperts, inter = cfg.moeIntermediateSize
        let sharedInter = cfg.sharedExpertIntermediateSize
        let topK = cfg.numExpertsPerTok

        var router = Self.linear(x, rows: S, inDim: D, weight: w.gate, outDim: E)
        var picksPerRow: [[(Int, Float)]] = []
        for s in 0..<S {
            Self.softmaxRow(&router, base: s * E, count: E)
            var picks: [(Int, Float)] = []
            for e in 0..<E {
                let p = router[s * E + e]
                if picks.count < topK {
                    picks.append((e, p))
                    picks.sort { $0.1 > $1.1 }
                } else if p > picks[topK - 1].1 {
                    picks[topK - 1] = (e, p)
                    picks.sort { $0.1 > $1.1 }
                }
            }
            picksPerRow.append(picks)
        }

        // Batch union: dequantize each selected expert exactly once for the
        // whole batch (same pattern the streaming runtime uses for prefill).
        var expertWeights: [Int: (g: [Float], u: [Float], d: [Float])] = [:]
        for picks in picksPerRow {
            for (e, _) in picks where expertWeights[e] == nil {
                expertWeights[e] = (
                    g: try ckpt.moduleWeightSlice(w.expertPrefix + "gate_proj", rowRange: e * inter..<(e + 1) * inter),
                    u: try ckpt.moduleWeightSlice(w.expertPrefix + "up_proj", rowRange: e * inter..<(e + 1) * inter),
                    d: try ckpt.moduleWeightSlice(w.expertPrefix + "down_proj", rowRange: e * D..<(e + 1) * D)
                )
            }
        }

        var out = [Float](repeating: 0, count: S * D)
        var gBuf = [Float](repeating: 0, count: inter)
        var uBuf = [Float](repeating: 0, count: inter)
        var dBuf = [Float](repeating: 0, count: D)
        var sharedG = [Float](repeating: 0, count: sharedInter)
        var sharedU = [Float](repeating: 0, count: sharedInter)

        for s in 0..<S {
            let picks = picksPerRow[s]
            var scoreSum: Float = 0
            for (_, p) in picks { scoreSum += p }

            for (e, p) in picks {
                let weight = cfg.normTopkProb ? p / scoreSum : p
                let ew = expertWeights[e]!
                Self.matvec(ew.g, 0, outDim: inter, inDim: D, x, s * D, into: &gBuf, 0)
                Self.matvec(ew.u, 0, outDim: inter, inDim: D, x, s * D, into: &uBuf, 0)
                for i in 0..<inter { gBuf[i] = Self.silu(gBuf[i]) * uBuf[i] }
                Self.matvec(ew.d, 0, outDim: D, inDim: inter, gBuf, 0, into: &dBuf, 0)
                for i in 0..<D { out[s * D + i] += weight * dBuf[i] }
            }

            Self.matvec(w.sharedGateProj, 0, outDim: sharedInter, inDim: D, x, s * D, into: &sharedG, 0)
            Self.matvec(w.sharedUpProj, 0, outDim: sharedInter, inDim: D, x, s * D, into: &sharedU, 0)
            for i in 0..<sharedInter { sharedG[i] = Self.silu(sharedG[i]) * sharedU[i] }
            Self.matvec(w.sharedDownProj, 0, outDim: D, inDim: sharedInter, sharedG, 0, into: &dBuf, 0)
            var sg: Float = 0
            for i in 0..<D { sg += w.sharedExpertGate[i] * x[s * D + i] }
            let sharedScale = Self.sigmoid(sg)
            for i in 0..<D { out[s * D + i] += sharedScale * dBuf[i] }
        }
        return out
    }
}
