import Foundation
import Metal
import os.signpost

/// GPU runtime: every linear (dense projections, router, experts, lm_head)
/// runs as a quantized GEMV directly on mmapped checkpoint bytes — weights are
/// never decompressed. On the fast path the full-attention layers run their
/// core on GPU too (S2): q/k prep + RoPE, KV append into a GPU-resident
/// cache, and gated causal softmax attention. The CPU keeps only glue —
/// top-k routing over GPU-computed router logits, expert-cache fetches,
/// embedding row copies, and the state.kv mirror append — while the
/// no-fast-path fallback still computes attention (norms, RoPE, softmax) on
/// CPU. Numerics mirror QwenCPUModel exactly.
public final class QwenMetalModel {
    /// S3b command-buffer phase labels. A label names the work a phase scope
    /// encoded; it claims nothing about when the GPU actually ran that work.
    public enum StepPhase: String, CaseIterable, Sendable {
        /// Attention norm + q/k/v projections, q/k prep + RoPE, KV append,
        /// causal softmax attention, output projection, residual add.
        case attention
        /// DeltaNet mixer: projections, conv step, recurrence, gated norm,
        /// output projection, residual add.
        case delta
        /// Routed expert GEMVs plus the shared-expert chain and accumulation.
        case moe
        /// Pre-MoE norm and the router logits projection.
        case router
        /// Final norm and vocabulary projection.
        case lmHead
        /// Dispatches encoded outside every labeled scope (must stay empty).
        case other
    }

    /// How real a per-phase GPU/wait split can be on this device without
    /// changing the schedule. The fast path deliberately encodes several
    /// phases into one encoder per command buffer, so splitting a shared
    /// buffer's GPU time by phase requires timestamp samples *inside* the
    /// encoder — dispatch-boundary counter sampling. Devices that sample only
    /// at encoder/stage boundaries (Apple GPUs) cannot provide that split for
    /// this schedule, and the API says so instead of inventing numbers.
    public enum PhaseGpuSplitSupport: Equatable, Sendable {
        /// The device samples GPU timestamps at compute dispatch boundaries;
        /// per-phase GPU time is measured inside shared buffers.
        case dispatchBoundaryCounters
        /// No in-encoder sampling on this device; per-phase GPU time is
        /// absent (nil), never zeros. The reason names the device and the
        /// missing capability.
        case unsupported(reason: String)
    }

    /// One committed command buffer, in commit order. Encode time and
    /// dispatch counts are attributed to phases exactly (encoding is CPU work
    /// under our control); blocking-wait time exists only at command-buffer
    /// granularity, and GPU time splits per phase only when the device
    /// supports dispatch-boundary counter sampling (see PhaseGpuSplitSupport)
    /// — otherwise a buffer spanning several phases cannot split it.
    public struct CommandBufferSample: Equatable, Sendable {
        /// Compute dispatches encoded into this buffer, per phase.
        public let phaseDispatches: [StepPhase: Int]
        /// CPU wall time inside each phase's encode scope for this buffer.
        public let phaseEncodeSeconds: [StepPhase: Double]
        /// CPU wall time from encoder creation to commit (>= the phase sum).
        public let encodeSeconds: Double
        /// Wall time inside this buffer's waitUntilCompleted call.
        public let waitSeconds: Double
        /// GPU start-to-end duration; nil when the buffer failed or reported
        /// invalid timestamps. Not proof of exclusive GPU occupancy.
        public let gpuSeconds: Double?
        /// GPU seconds per phase from dispatch-boundary counter samples
        /// resolved for this buffer. nil when the device cannot sample inside
        /// an encoder, or when any of this buffer's samples failed to resolve
        /// — a partial split would silently under-report a phase.
        public let phaseGpuSeconds: [StepPhase: Double]?
        /// Whether status was .completed after the blocking wait.
        public let completed: Bool

        /// Phases encoded into this buffer, in declaration order (a canonical
        /// label, not the encode order within the buffer).
        public var phases: [StepPhase] {
            StepPhase.allCases.filter {
                phaseDispatches[$0] != nil || phaseEncodeSeconds[$0] != nil
            }
        }

        public var dispatchesEncoded: Int { phaseDispatches.values.reduce(0, +) }
    }

    /// S3a whole-step aggregates plus the S3b committed-buffer timeline.
    /// The aggregates still cannot identify overlap; the timeline labels each
    /// buffer's encoded phases and reports per-buffer encode/wait/GPU cost,
    /// but it is not an Instruments trace: it cannot see gaps between buffers
    /// or split a multi-phase buffer's wait/GPU time by phase.
    public struct StepMetrics: Equatable, Sendable {
        public let tokensProcessed: Int
        public let logitProjections: Int
        public let commandBuffersCommitted: Int
        public let blockingWaits: Int
        /// Wall time spent inside the existing waitUntilCompleted calls.
        public let blockingWaitSeconds: Double
        /// Compute dispatch calls encoded, including partial work if step throws.
        public let computeDispatchesEncoded: Int
        public let stepWallSeconds: Double
        /// Buffers whose status was not completed after waitUntilCompleted.
        public let commandBufferErrors: Int
        /// Sum of valid per-command-buffer GPU durations, not an elapsed span.
        public let gpuExecutionSeconds: Double
        public let gpuTimedCommandBuffers: Int
        /// Completed buffers with unavailable or invalid GPU timestamps.
        public let gpuUntimedCommandBuffers: Int
        /// False when step exited by throwing after publishing partial counters.
        public let completedWithoutThrow: Bool
        /// S3b: committed command buffers in commit order. A throwing step's
        /// never-committed encodes are absent here but still counted in
        /// computeDispatchesEncoded and the phase totals below.
        public let commandBufferTimeline: [CommandBufferSample]
        /// Encode-side dispatch totals per phase, including partial work from
        /// a buffer the step never committed before throwing.
        public let phaseDispatchesEncoded: [StepPhase: Int]
        /// CPU wall time inside per-phase encode scopes; same partial-work
        /// semantics as phaseDispatchesEncoded.
        public let phaseEncodeSeconds: [StepPhase: Double]
        /// GPU seconds per phase, summed over the buffers whose
        /// dispatch-boundary counter samples resolved. nil whenever the
        /// device cannot sample inside an encoder (phaseGpuSplitSupport is
        /// .unsupported) — absent, never zeros that look like measurements.
        public let phaseGpuSeconds: [StepPhase: Double]?
        /// S3c: sub-attribution of the CPU gap (wall − wait − encode) to the
        /// CPU work the step performs between buffers; see CPUGapBreakdown.
        public let cpuGap: CPUGapBreakdown

        public var avoidedLogitProjections: Int {
            max(0, tokensProcessed - logitProjections)
        }
    }

    /// Count of os_signpost intervals one step emitted, per interval name.
    /// Rebuilt per step; exists so tests can pin emission to the timeline.
    struct SignpostTally: Equatable {
        var stepIntervals = 0
        var phaseIntervals = 0
        var commandBufferIntervals = 0
    }

    private final class StepCounters {
        var tokensProcessed = 0
        var logitProjections = 0
        var commandBuffersCommitted = 0
        var blockingWaits = 0
        var blockingWaitSeconds = 0.0
        var computeDispatchesEncoded = 0
        var commandBufferErrors = 0
        var gpuExecutionSeconds = 0.0
        var gpuTimedCommandBuffers = 0
        var gpuUntimedCommandBuffers = 0
        var completedWithoutThrow = false
        var timeline: [CommandBufferSample] = []
        var phaseDispatches: [StepPhase: Int] = [:]
        var phaseEncodeSeconds: [StepPhase: Double] = [:]
        /// Step totals of resolved per-phase GPU seconds; nil unless the
        /// device supports dispatch-boundary sampling.
        var phaseGpuSeconds: [StepPhase: Double]?
        /// S3c gap scopes (see CPUGapBreakdown); flat, disjoint, throw-safe.
        var gapEmbeddingSeconds = 0.0
        var gapEmbeddingLookups = 0
        var gapRouterSeconds = 0.0
        var gapExpertFetchSeconds = 0.0
        var gapExpertFetchHits = 0
        var gapExpertFetchMisses = 0
        var gapKVMirrorSeconds = 0.0
        var gapCommandBufferSetupSeconds = 0.0
        var gapCommitSeconds = 0.0
        var gapLogitsReadbackSeconds = 0.0
        var currentPhase: StepPhase?
        var bufferOpen = false
        var bufferEncodeStart = 0.0
        var bufferPhaseDispatches: [StepPhase: Int] = [:]
        var bufferPhaseEncodeSeconds: [StepPhase: Double] = [:]
        /// Encoder of the open buffer, for in-encoder counter samples only.
        var currentEncoder: MTLComputeCommandEncoder?
        var bufferSampleBuffer: MTLCounterSampleBuffer?
        var bufferSampleIndex = 0
        var bufferPhaseSampleRanges: [(phase: StepPhase, start: Int, end: Int)] = []
        var bufferSamplingBroken = false
        /// CPU/GPU correlation captured when the open buffer began encoding.
        var bufferCorrelationStart: (cpu: MTLTimestamp, gpu: MTLTimestamp)?
        var signpostTally = SignpostTally()
    }

    public let config: QwenConfig
    let ckpt: Checkpoint
    let engine: MetalEngine
    let store: MetalShardStore
    typealias GPULinear = MetalShardStore.GPULinear

    struct ExpertStack {
        var lin: GPULinear
        var wStride: Int   // per-expert advance, bytes
        var sStride: Int   // per-expert advance, bytes
        var rows: Int      // rows per expert
    }

    struct AttnGPU {
        var qProj, kProj, vProj, oProj: GPULinear
        var qNorm, kNorm: [Float]
    }

    struct DeltaGPU {
        var qkvz, ba, qkv, zProj, bProj, aProj: GPULinear?
        var conv, dtBias, aLog, norm: [Float]
        var outProj: GPULinear
    }

    struct MoEGPU {
        var gate: GPULinear
        var stacks: (gate: ExpertStack, up: ExpertStack, down: ExpertStack)?
        var sharedGate, sharedUp, sharedDown: GPULinear
        var sharedExpertGate: [Float]
    }

    /// Per-expert projection geometry inside a .qpack blob (offsets in bytes).
    struct ExpertProj {
        var wOff, sOff, bOff: Int
        var outDim, inDim: Int
        var groupSize, bits: Int
        var scalesType: UInt32
        var isQuantized: Bool
        var plainDtype: UInt32

        func linear(on buf: MTLBuffer) -> GPULinear {
            GPULinear(
                wBuffer: buf, sBuffer: buf, bBuffer: buf,
                outDim: outDim, inDim: inDim, isQuantized: isQuantized,
                groupSize: groupSize, bits: bits, scalesType: scalesType,
                wOff: wOff, sOff: sOff, bOff: bOff, plainDtype: plainDtype
            )
        }
    }

    struct LayerGPU {
        var inputNorm, postAttnNorm: [Float]
        var attn: AttnGPU?
        var delta: DeltaGPU?
        var moe: MoEGPU
    }

    var layers: [LayerGPU] = []
    var finalNorm: [Float]
    var lmHead: GPULinear
    public internal(set) var expertCache: ExpertCache?
    var expertProjs: (gate: ExpertProj, up: ExpertProj, down: ExpertProj)?
    let xBuf: MTLBuffer
    let yBuf: MTLBuffer
    let logitsBuf: MTLBuffer
    /// Latest step snapshot. A throwing step publishes its partial counters.
    public private(set) var lastStepMetrics = StepMetrics(
        tokensProcessed: 0, logitProjections: 0,
        commandBuffersCommitted: 0, blockingWaits: 0, blockingWaitSeconds: 0,
        computeDispatchesEncoded: 0, stepWallSeconds: 0, commandBufferErrors: 0,
        gpuExecutionSeconds: 0, gpuTimedCommandBuffers: 0, gpuUntimedCommandBuffers: 0,
        completedWithoutThrow: false,
        commandBufferTimeline: [], phaseDispatchesEncoded: [:], phaseEncodeSeconds: [:],
        phaseGpuSeconds: nil, cpuGap: .zero
    )
    private var activeStepCounters: StepCounters?
    /// Whether a per-phase GPU split is real on this device, decided by the
    /// runtime counter probe at init — never assumed from the OS or GPU name.
    public let phaseGpuSplitSupport: PhaseGpuSplitSupport
    /// The raw probe behind phaseGpuSplitSupport, for reporting.
    public var counterSamplingSupport: MetalEngine.CounterSamplingSupport {
        engine.probeCounterSampling()
    }
    /// Signpost intervals emitted by the latest step; mirrors the timeline.
    internal private(set) var lastSignpostTally = SignpostTally()
    /// os_signpost log for step/phase/commandBuffer intervals. Instruments'
    /// os_signpost instrument groups them under subsystem "Swiftlet".
    private static let signpostLog = OSLog(subsystem: "Swiftlet", category: "MetalStep")
    /// Sample slots per command buffer: the fast path encodes at most three
    /// phase scopes per buffer (2 samples each); headroom is harmless.
    private static let maxPhaseSamplesPerBuffer = 16
    /// Test hook (S1b-a): observes the routed expert list for every
    /// (token, layer) exactly as the current schedule selects it — token
    /// order in the token-major paths, layer-major order (per layer, tokens
    /// ascending) in the S1b prefill. The routes themselves are identical
    /// either way; only the visit order differs. Stays internal.
    var routedExpertObserver: ((_ layer: Int, _ experts: [Int]) -> Void)?
    /// Test hook (S1b): the expert union the layer-major prefill actually
    /// fetched and encoded for one (chunk, layer), ascending expert order —
    /// what the PrefillExpertUnionPlan oracle predicts. Stays internal.
    var prefillExpertUnionObserver: ((_ layer: Int, _ experts: [Int]) -> Void)?

    /// Prompt-prefill schedule for multi-token step calls. Single-token
    /// decode steps never consult this.
    public enum PrefillMode: Equatable, Sendable {
        /// Legacy S1a schedule: the decode schedule repeated per token, with
        /// intermediate LM heads elided.
        case tokenMajor
        /// S1b schedule: split the prompt into chunks of at most chunkTokens;
        /// inside a chunk, sweep layer-by-layer across all tokens and touch
        /// each layer's expert union once per chunk. chunkTokens bounds
        /// scratch memory: the chunk keeps chunkTokens x Regions.total
        /// scratch floats and chunkTokens hidden rows resident.
        case layerMajor(chunkTokens: Int)
    }

    /// Default: layer-major with 32-token chunks. One decode-shaped buffer
    /// sequence per chunk instead of per token, and expert fetches per (layer, union)
    /// instead of per (token, layer, pick). 32 keeps scratch bounded to a few
    /// tens of MB on 80B-class configs while letting short prompts run as a
    /// single chunk; unions saturate toward the full expert set beyond a few
    /// dozen tokens, so larger chunks add memory faster than they add
    /// expert-traffic savings. `.tokenMajor` restores the S1a schedule.
    public var prefillMode: PrefillMode = .layerMajor(chunkTokens: 32)

    // MARK: Fast path (split DeltaNet layout): one command buffer per layer.
    struct Regions {
        var x0 = 0, qkv = 0, z = 0, b = 0, a = 0, conv = 0, gb = 0
        var dy = 0, dn = 0, r = 0, xmoe = 0, rout = 0
        var qout = 0, knew = 0, vnew = 0, att = 0, qprep = 0
        var qkvzStage = 0, baStage = 0
        var exp = 0, dexp = 0, sh = 0, shg = 0
        var total = 0
    }
    struct FastLayer {
        var inputNorm, postNorm: MTLBuffer
        var convW, aLog, dtBias, deltaNormW: MTLBuffer?
        var hist, state: MTLBuffer?
        var sharedGateLin: GPULinear
        // S2, attention layers only: q/k norm weights and the GPU-resident
        // KV cache ([position][kvHead][headDim] rows, grown on demand). Rows
        // are indexed by absolute position, so a fresh state simply
        // overwrites from row 0; rows past the current position are never
        // read.
        var qNormW, kNormW: MTLBuffer?
        var kCache, vCache: MTLBuffer?
        var kvCapacity = 0
    }
    var fastLayers: [FastLayer] = []
    var finalNormBuf: MTLBuffer?
    var sBuf: MTLBuffer?
    var hBuf: MTLBuffer?
    var reg = Regions()
    var boundStateID: ObjectIdentifier?
    struct PendingMoE {
        var bufs: [MTLBuffer]
        var weights: [Float]
        var stacksLayer: Int
        var picks: [(Int, Float)]
    }

    // S1b layer-major prefill scratch: one Regions stride and one hidden row
    // per chunk token. Allocated lazily on the first layer-major prompt call
    // and kept for the model's lifetime; failure to allocate falls back to
    // the token-major schedule instead of crashing.
    var prefillScratchBuf: MTLBuffer?
    var prefillHiddenBuf: MTLBuffer?
    var prefillSlotCapacity = 0
    var prefillStage = PrefillStage()

    public init(modelDir: URL, cacheBudgetGB: Double = 8) throws {
        config = try QwenConfig(url: modelDir.appendingPathComponent("config.json"))
        ckpt = try Checkpoint(dir: modelDir)
        engine = try MetalEngine()
        store = MetalShardStore(device: engine.device)
        phaseGpuSplitSupport = Self.probePhaseGpuSplit(engine: engine)

        let cfg = config
        let ckpt = self.ckpt
        let store = self.store

        // .qpack container: dense weights copied into resident GPU buffers,
        // experts streamed through the bounded cache. Raw checkpoint dir:
        // everything mmapped (fine when the model fits in RAM).
        let qpackMode = FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("packed_experts/layout.json").path
        )
        func denseLinear(_ path: String) throws -> GPULinear {
            qpackMode
                ? try store.residentLinear(ckpt: ckpt, path: path)
                : try store.gpuLinear(ckpt: ckpt, path: path)
        }

        finalNorm = try ckpt.tensor("model.norm.weight")
        lmHead = try denseLinear(cfg.tieWordEmbeddings ? "model.embed_tokens" : "lm_head")

        if qpackMode {
            let cache = try ExpertCache(
                containerDir: modelDir, device: engine.device,
                budgetBytes: Int(cacheBudgetGB * 1_073_741_824)
            )
            expertCache = cache

            let manifestData = try Data(contentsOf: modelDir.appendingPathComponent("manifest.json"))
            let manifest = try JSONDecoder().decode(Qpack.Manifest.self, from: manifestData)
            func proj(_ name: String, outDim: Int) throws -> ExpertProj {
                guard let w = cache.reader.section(name + ".weight") else {
                    throw Checkpoint.Error.missingTensor("qpack section \(name)")
                }
                if w.dtype == "U32", let bits = manifest.quantBits, let group = manifest.quantGroupSize,
                   let sSec = cache.reader.section(name + ".scales"),
                   let bSec = cache.reader.section(name + ".biases") {
                    let inDim = try Qpack.expertLogicalInDim(
                        weightLastDim: w.shape.last ?? 0, scalesLastDim: sSec.shape.last ?? 0,
                        bits: bits, groupSize: group, section: name)
                    return ExpertProj(
                        wOff: w.offset, sOff: sSec.offset, bOff: bSec.offset,
                        outDim: outDim, inDim: inDim,
                        groupSize: group, bits: bits,
                        scalesType: MetalEngine.ScalesType(dtype: sSec.dtype)?.rawValue ?? 2,
                        isQuantized: true, plainDtype: 0
                    )
                }
                let dtype: UInt32 = w.dtype == "F32" ? 0 : (w.dtype == "F16" ? 1 : 2)
                return ExpertProj(
                    wOff: w.offset, sOff: 0, bOff: 0,
                    outDim: outDim, inDim: w.shape.last!,
                    groupSize: 0, bits: 0, scalesType: 0,
                    isQuantized: false, plainDtype: dtype
                )
            }
            expertProjs = (
                gate: try proj("gate_proj", outDim: cfg.moeIntermediateSize),
                up: try proj("up_proj", outDim: cfg.moeIntermediateSize),
                down: try proj("down_proj", outDim: cfg.hiddenSize)
            )
        }
        func expertStack(_ path: String, rowsPerExpert: Int) throws -> ExpertStack {
            let lin = try store.gpuLinear(ckpt: ckpt, path: path)
            if lin.isQuantized {
                let perWord = 32 / lin.bits
                let packedCols = lin.inDim / perWord
                let groups = lin.inDim / lin.groupSize
                let scaleBytes = lin.scalesType == 0 ? 4 : 2
                return ExpertStack(
                    lin: lin,
                    wStride: rowsPerExpert * packedCols * 4,      // bytes
                    sStride: rowsPerExpert * groups * scaleBytes, // bytes
                    rows: rowsPerExpert
                )
            }
            let elemBytes = lin.plainDtype == 0 ? 4 : 2
            return ExpertStack(
                lin: lin,
                wStride: rowsPerExpert * lin.inDim * elemBytes,
                sStride: 0,
                rows: rowsPerExpert
            )
        }

        for i in 0..<cfg.numHiddenLayers {
            let p = "model.layers.\(i)."
            let stacks: (ExpertStack, ExpertStack, ExpertStack)? = qpackMode ? nil : (
                try expertStack(p + "mlp.switch_mlp.gate_proj", rowsPerExpert: cfg.moeIntermediateSize),
                try expertStack(p + "mlp.switch_mlp.up_proj", rowsPerExpert: cfg.moeIntermediateSize),
                try expertStack(p + "mlp.switch_mlp.down_proj", rowsPerExpert: cfg.hiddenSize)
            )
            let moe = MoEGPU(
                gate: try denseLinear(p + "mlp.gate"),
                stacks: stacks,
                sharedGate: try denseLinear(p + "mlp.shared_expert.gate_proj"),
                sharedUp: try denseLinear(p + "mlp.shared_expert.up_proj"),
                sharedDown: try denseLinear(p + "mlp.shared_expert.down_proj"),
                sharedExpertGate: try ckpt.moduleWeight(p + "mlp.shared_expert_gate")
            )
            var layer = LayerGPU(
                inputNorm: try ckpt.tensor(p + "input_layernorm.weight"),
                postAttnNorm: try ckpt.tensor(p + "post_attention_layernorm.weight"),
                moe: moe
            )
            if cfg.isLinearLayer(i) {
                var d = DeltaGPU(
                    conv: try ckpt.tensor(p + "linear_attn.conv1d.weight"),
                    dtBias: try ckpt.tensor(p + "linear_attn.dt_bias"),
                    aLog: try ckpt.tensor(p + "linear_attn.A_log"),
                    norm: try ckpt.tensor(p + "linear_attn.norm.weight"),
                    outProj: try denseLinear(p + "linear_attn.out_proj")
                )
                switch cfg.deltaLayout {
                case .fusedInterleaved:
                    d.qkvz = try denseLinear(p + "linear_attn.in_proj_qkvz")
                    d.ba = try denseLinear(p + "linear_attn.in_proj_ba")
                case .split:
                    d.qkv = try denseLinear(p + "linear_attn.in_proj_qkv")
                    d.zProj = try denseLinear(p + "linear_attn.in_proj_z")
                    d.bProj = try denseLinear(p + "linear_attn.in_proj_b")
                    d.aProj = try denseLinear(p + "linear_attn.in_proj_a")
                }
                layer.delta = d
            } else {
                layer.attn = AttnGPU(
                    qProj: try denseLinear(p + "self_attn.q_proj"),
                    kProj: try denseLinear(p + "self_attn.k_proj"),
                    vProj: try denseLinear(p + "self_attn.v_proj"),
                    oProj: try denseLinear(p + "self_attn.o_proj"),
                    qNorm: try ckpt.tensor(p + "self_attn.q_norm.weight"),
                    kNorm: try ckpt.tensor(p + "self_attn.k_norm.weight")
                )
            }
            layers.append(layer)
        }

        let maxIn = max(cfg.hiddenSize, cfg.valueDim, cfg.numAttentionHeads * cfg.headDim, cfg.convDim)
        xBuf = engine.device.makeBuffer(length: maxIn * 4, options: .storageModeShared)!
        let moeScratch = 3 * cfg.numExpertsPerTok * cfg.moeIntermediateSize
            + cfg.numExpertsPerTok * cfg.hiddenSize
            + 3 * cfg.sharedExpertIntermediateSize + cfg.hiddenSize
            + cfg.numExperts + 64
        let yLen = max(
            2 * cfg.keyDim + 2 * cfg.valueDim + 2 * cfg.linearNumValueHeads + 64,
            moeScratch,
            2 * cfg.numAttentionHeads * cfg.headDim + 2 * cfg.numKeyValueHeads * cfg.headDim + 64
        )
        yBuf = engine.device.makeBuffer(length: yLen * 4, options: .storageModeShared)!
        logitsBuf = engine.device.makeBuffer(length: cfg.vocabSize * 4, options: .storageModeShared)!

        print("[QwenMetalModel] descriptors ready; building fast path...")
        try setupFastPath(denseLinear: denseLinear)
        print("[QwenMetalModel] fast path ready")
    }

    private func setupFastPath(denseLinear: (String) throws -> GPULinear) throws {
        let cfg = config
        let D = cfg.hiddenSize, E = cfg.numExperts, nv = cfg.linearNumValueHeads
        let H = cfg.numAttentionHeads, hd = cfg.headDim, KVH = cfg.numKeyValueHeads
        let inter = cfg.moeIntermediateSize, shInter = cfg.sharedExpertIntermediateSize
        let K = cfg.numExpertsPerTok

        var o = Regions()
        var c = 0
        func take(_ n: Int) -> Int { let v = c; c += n; return v }
        o.x0 = take(D)
        o.qkv = take(cfg.convDim)
        o.z = take(cfg.valueDim)
        o.b = take(nv)
        o.a = take(nv)
        o.conv = take(cfg.convDim)
        o.gb = take(2 * nv)
        o.dy = take(cfg.valueDim)
        o.dn = take(cfg.valueDim)
        o.r = take(D)
        o.xmoe = take(D)
        o.rout = take(E)
        o.qout = take(2 * H * hd)
        o.knew = take(KVH * hd)
        o.vnew = take(KVH * hd)
        o.att = take(H * hd)
        o.qprep = take(H * hd)
        o.qkvzStage = take(2 * cfg.keyDim + 2 * cfg.valueDim)
        o.baStage = take(2 * nv)
        o.exp = take(3 * K * inter)
        o.dexp = take(K * D)
        o.sh = take(3 * shInter + D)
        o.shg = take(4)
        o.total = c
        reg = o

        let opts: MTLResourceOptions = [.storageModeShared, .hazardTrackingModeUntracked]
        sBuf = engine.device.makeBuffer(length: o.total * 4, options: opts)
        hBuf = engine.device.makeBuffer(length: D * 4, options: opts)
        finalNormBuf = engine.makeBuffer(finalNorm)

        for i in 0..<cfg.numHiddenLayers {
            let L = layers[i]
            let p = "model.layers.\(i)."
            var fl = FastLayer(
                inputNorm: engine.makeBuffer(L.inputNorm),
                postNorm: engine.makeBuffer(L.postAttnNorm),
                sharedGateLin: try denseLinear(p + "mlp.shared_expert_gate")
            )
            if let d = L.delta {
                fl.convW = engine.makeBuffer(d.conv)
                fl.aLog = engine.makeBuffer(d.aLog)
                fl.dtBias = engine.makeBuffer(d.dtBias)
                fl.deltaNormW = engine.makeBuffer(d.norm)
                fl.hist = engine.device.makeBuffer(length: (cfg.linearConvKernelDim - 1) * cfg.convDim * 4, options: opts)
                fl.state = engine.device.makeBuffer(
                    length: nv * cfg.linearValueHeadDim * cfg.linearKeyHeadDim * 4, options: opts)
            }
            if let a = L.attn {
                fl.qNormW = engine.makeBuffer(a.qNorm)
                fl.kNormW = engine.makeBuffer(a.kNorm)
                // kCache/vCache grow on demand in ensureKVCapacity.
            }
            fastLayers.append(fl)
        }
    }

    /// Replaces the expert cache with a smaller one (old slots free
    /// immediately; the new cache refills lazily). Memory-pressure valve.
    public func shrinkCache(toGB gb: Double) {
        guard expertCache != nil else { return }
        // Int(Double) traps on NaN, infinity, and out-of-range values; a
        // pressure valve must refuse such a request, not crash on it.
        let bytes = gb * 1_073_741_824
        guard bytes >= 0, let budget = Int(exactly: bytes.rounded(.down)) else { return }
        guard let replacement = try? ExpertCache(
            containerDir: ckpt.dir, device: engine.device, budgetBytes: budget
        ) else { return }
        expertCache = replacement
    }

    // MARK: - GPU phase helper

    private func loadX(_ v: [Float]) {
        v.withUnsafeBufferPointer {
            xBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: v.count * 4)
        }
    }

    /// Decides whether a per-phase GPU split is measurable here. The frozen
    /// schedule shares one encoder across phases, so the split needs
    /// dispatch-boundary sampling; anything less gets an explicit reason.
    private static func probePhaseGpuSplit(engine: MetalEngine) -> PhaseGpuSplitSupport {
        let probe = engine.probeCounterSampling()
        if probe.atDispatchBoundary && probe.hasTimestampCounterSet {
            return .dispatchBoundaryCounters
        }
        if !probe.hasTimestampCounterSet {
            return .unsupported(reason:
                "device \(probe.deviceName) exposes no timestamp counter set")
        }
        return .unsupported(reason:
            "device \(probe.deviceName) samples timestamps only at encoder boundaries"
            + " (stage=\(probe.atStageBoundary)), and the fast-path schedule encodes"
            + " several phases into one encoder per command buffer; an in-buffer"
            + " per-phase GPU split cannot be measured without changing the schedule")
    }

    /// Starts a step command buffer and opens its S3b timeline accumulator so
    /// encode time and dispatches attribute to this buffer until commit. When
    /// the device supports dispatch-boundary sampling, a fresh counter sample
    /// buffer and a CPU/GPU correlation point are attached for the per-phase
    /// GPU split; on other devices this adds nothing to the hot path.
    private func beginStepCommandBuffer() -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
        let setupStart = ProcessInfo.processInfo.systemUptime
        let cb = engine.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        if let counters = activeStepCounters {
            counters.gapCommandBufferSetupSeconds +=
                max(0, ProcessInfo.processInfo.systemUptime - setupStart)
            counters.bufferOpen = true
            counters.bufferEncodeStart = ProcessInfo.processInfo.systemUptime
            counters.bufferPhaseDispatches = [:]
            counters.bufferPhaseEncodeSeconds = [:]
            counters.currentEncoder = enc
            counters.bufferSampleBuffer = nil
            counters.bufferSampleIndex = 0
            counters.bufferPhaseSampleRanges = []
            counters.bufferSamplingBroken = false
            counters.bufferCorrelationStart = nil
            if case .dispatchBoundaryCounters = phaseGpuSplitSupport,
               let tsSet = engine.timestampCounterSet {
                let desc = MTLCounterSampleBufferDescriptor()
                desc.counterSet = tsSet
                desc.storageMode = .shared
                desc.sampleCount = Self.maxPhaseSamplesPerBuffer
                counters.bufferSampleBuffer =
                    try? engine.device.makeCounterSampleBuffer(descriptor: desc)
                counters.bufferSamplingBroken = counters.bufferSampleBuffer == nil
                counters.bufferCorrelationStart = engine.device.sampleTimestamps()
            }
        }
        return (cb, enc)
    }

    /// Attributes encode-side work (CPU encode wall time and dispatch counts)
    /// to a phase label, emits one os_signpost interval per scope, and — on
    /// devices with dispatch-boundary sampling — brackets the scope with GPU
    /// timestamp samples. Scopes never nest in the current schedule; if one
    /// ever did, its elapsed time would double-count, so keep call sites flat.
    private func phase<T>(_ p: StepPhase, _ body: () throws -> T) rethrows -> T {
        guard let counters = activeStepCounters else { return try body() }
        let previous = counters.currentPhase
        counters.currentPhase = p
        let signpostID = OSSignpostID(log: Self.signpostLog)
        os_signpost(.begin, log: Self.signpostLog, name: "phase",
                    signpostID: signpostID, "%{public}@", p.rawValue)
        counters.signpostTally.phaseIntervals += 1
        var sampleStart = -1
        if let sb = counters.bufferSampleBuffer, counters.bufferOpen,
           let enc = counters.currentEncoder, !counters.bufferSamplingBroken {
            if counters.bufferSampleIndex + 1 < Self.maxPhaseSamplesPerBuffer {
                sampleStart = counters.bufferSampleIndex
                counters.bufferSampleIndex += 2
                enc.sampleCounters(sampleBuffer: sb, sampleIndex: sampleStart, barrier: true)
            } else {
                // Out of slots: drop the whole buffer's split rather than
                // publish a partial one.
                counters.bufferSamplingBroken = true
            }
        }
        let start = ProcessInfo.processInfo.systemUptime
        defer {
            let elapsed = max(0, ProcessInfo.processInfo.systemUptime - start)
            counters.phaseEncodeSeconds[p, default: 0] += elapsed
            if counters.bufferOpen {
                counters.bufferPhaseEncodeSeconds[p, default: 0] += elapsed
            }
            if sampleStart >= 0, let sb = counters.bufferSampleBuffer,
               let enc = counters.currentEncoder, !counters.bufferSamplingBroken {
                enc.sampleCounters(sampleBuffer: sb, sampleIndex: sampleStart + 1, barrier: true)
                counters.bufferPhaseSampleRanges.append(
                    (phase: p, start: sampleStart, end: sampleStart + 1))
            }
            os_signpost(.end, log: Self.signpostLog, name: "phase",
                        signpostID: signpostID, "%{public}@", p.rawValue)
            counters.currentPhase = previous
        }
        return try body()
    }

    /// S3c: attributes `body`'s wall time to one CPU-gap scope. Scopes are
    /// flat by construction (none of the call sites nest); a nested call
    /// would double-count, so keep them that way.
    private func gapScope<T>(
        _ field: ReferenceWritableKeyPath<StepCounters, Double>, _ body: () throws -> T
    ) rethrows -> T {
        guard let counters = activeStepCounters else { return try body() }
        let start = ProcessInfo.processInfo.systemUptime
        defer {
            counters[keyPath: field] += max(0, ProcessInfo.processInfo.systemUptime - start)
        }
        return try body()
    }

    /// Embedding row for one token, timed as the embedding gap scope.
    private func embeddingRow(_ token: Int) throws -> [Float] {
        try gapScope(\.gapEmbeddingSeconds) {
            activeStepCounters?.gapEmbeddingLookups += 1
            return try ckpt.moduleWeightSlice("model.embed_tokens", rowRange: token..<(token + 1))
        }
    }

    /// Expert-cache fetch timed as the fetch gap scope, with the cache's own
    /// hit/miss counters differenced across the call.
    private func fetchExperts(_ cache: ExpertCache, layer: Int, experts: [Int]) throws -> [MTLBuffer] {
        try gapScope(\.gapExpertFetchSeconds) {
            let hits0 = cache.hits, misses0 = cache.misses
            defer {
                activeStepCounters?.gapExpertFetchHits += cache.hits - hits0
                activeStepCounters?.gapExpertFetchMisses += cache.misses - misses0
            }
            return try cache.buffers(layer: layer, experts: experts)
        }
    }

    /// Vocabulary logits copied out of the shared buffer, timed as the
    /// logits-readback gap scope.
    private func readLogits() -> [Float] {
        gapScope(\.gapLogitsReadbackSeconds) {
            Array(UnsafeBufferPointer(
                start: logitsBuf.contents().bindMemory(to: Float.self, capacity: config.vocabSize),
                count: config.vocabSize))
        }
    }

    private func commitAndWait(_ cb: MTLCommandBuffer) {
        let counters = activeStepCounters
        var encodeSeconds = 0.0
        if let counters, counters.bufferOpen {
            encodeSeconds = max(
                0, ProcessInfo.processInfo.systemUptime - counters.bufferEncodeStart
            )
        }
        let signpostID = OSSignpostID(log: Self.signpostLog)
        if counters != nil {
            os_signpost(.begin, log: Self.signpostLog, name: "commandBuffer", signpostID: signpostID)
            counters?.signpostTally.commandBufferIntervals += 1
        }
        let commitStart = ProcessInfo.processInfo.systemUptime
        cb.commit()
        counters?.commandBuffersCommitted += 1
        let waitStart = ProcessInfo.processInfo.systemUptime
        counters?.gapCommitSeconds += max(0, waitStart - commitStart)
        cb.waitUntilCompleted()
        counters?.blockingWaits += 1
        let waitSeconds = max(0, ProcessInfo.processInfo.systemUptime - waitStart)
        counters?.blockingWaitSeconds += waitSeconds

        let completed = cb.status == .completed
        var gpuSeconds: Double?
        if !completed {
            counters?.commandBufferErrors += 1
        } else {
            let start = cb.gpuStartTime
            let end = cb.gpuEndTime
            if start.isFinite, end.isFinite, start > 0, end > 0, end >= start {
                gpuSeconds = end - start
                counters?.gpuExecutionSeconds += end - start
                counters?.gpuTimedCommandBuffers += 1
            } else {
                counters?.gpuUntimedCommandBuffers += 1
            }
        }
        guard let counters else { return }

        var phaseGpu: [StepPhase: Double]?
        if completed, let sb = counters.bufferSampleBuffer,
           !counters.bufferSamplingBroken, !counters.bufferPhaseSampleRanges.isEmpty,
           let correlationStart = counters.bufferCorrelationStart {
            phaseGpu = resolvePhaseGpuSeconds(
                sampleBuffer: sb, sampleCount: counters.bufferSampleIndex,
                ranges: counters.bufferPhaseSampleRanges,
                correlationStart: correlationStart
            )
            if let resolved = phaseGpu, counters.phaseGpuSeconds != nil {
                for (p, s) in resolved {
                    counters.phaseGpuSeconds![p, default: 0] += s
                }
            }
        }

        let label = StepPhase.allCases.filter {
            counters.bufferPhaseDispatches[$0] != nil
                || counters.bufferPhaseEncodeSeconds[$0] != nil
        }.map(\.rawValue).joined(separator: "+")
        os_signpost(.end, log: Self.signpostLog, name: "commandBuffer",
                    signpostID: signpostID, "%{public}@ wait=%.6f gpu=%.6f",
                    label, waitSeconds, gpuSeconds ?? -1)

        counters.timeline.append(CommandBufferSample(
            phaseDispatches: counters.bufferPhaseDispatches,
            phaseEncodeSeconds: counters.bufferPhaseEncodeSeconds,
            encodeSeconds: encodeSeconds,
            waitSeconds: waitSeconds,
            gpuSeconds: gpuSeconds,
            phaseGpuSeconds: phaseGpu,
            completed: completed
        ))
        counters.bufferOpen = false
        counters.bufferPhaseDispatches = [:]
        counters.bufferPhaseEncodeSeconds = [:]
        counters.currentEncoder = nil
        counters.bufferSampleBuffer = nil
        counters.bufferSampleIndex = 0
        counters.bufferPhaseSampleRanges = []
        counters.bufferSamplingBroken = false
        counters.bufferCorrelationStart = nil
    }

    /// Converts resolved dispatch-boundary timestamp samples into per-phase
    /// GPU seconds using a linear CPU/GPU correlation across the buffer's
    /// lifetime (device.sampleTimestamps at encode start and now; CPU
    /// timestamps are nanoseconds). Any unresolved or non-monotone sample
    /// voids the whole buffer's split — partial splits under-report silently.
    /// Only reachable on devices whose probe reports dispatch-boundary
    /// sampling; Apple GPUs never enter here.
    private func resolvePhaseGpuSeconds(
        sampleBuffer: MTLCounterSampleBuffer,
        sampleCount: Int,
        ranges: [(phase: StepPhase, start: Int, end: Int)],
        correlationStart: (cpu: MTLTimestamp, gpu: MTLTimestamp)
    ) -> [StepPhase: Double]? {
        let correlationEnd = engine.device.sampleTimestamps()
        guard correlationEnd.cpu > correlationStart.cpu,
              correlationEnd.gpu > correlationStart.gpu else { return nil }
        let cpuSeconds = Double(correlationEnd.cpu - correlationStart.cpu) / 1e9
        let ticksPerSecond = Double(correlationEnd.gpu - correlationStart.gpu) / cpuSeconds
        guard ticksPerSecond > 0,
              let data = try? sampleBuffer.resolveCounterRange(0..<sampleCount),
              data.count >= sampleCount * MemoryLayout<MTLCounterResultTimestamp>.stride
        else { return nil }
        return data.withUnsafeBytes { raw -> [StepPhase: Double]? in
            let stamps = raw.bindMemory(to: MTLCounterResultTimestamp.self)
            var result: [StepPhase: Double] = [:]
            for range in ranges {
                let t0 = stamps[range.start].timestamp
                let t1 = stamps[range.end].timestamp
                guard t0 != 0, t1 != 0, t0 != .max, t1 != .max, t1 >= t0 else {
                    return nil
                }
                result[range.phase, default: 0] += Double(t1 - t0) / ticksPerSecond
            }
            return result
        }
    }

    private func runPhase(
        _ label: StepPhase, _ body: (MTLComputeCommandEncoder) throws -> Void
    ) throws {
        let (cb, enc) = beginStepCommandBuffer()
        try phase(label) { try body(enc) }
        enc.endEncoding()
        commitAndWait(cb)
    }

    private func readY(_ offset: Int, _ count: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: yBuf.contents().advanced(by: offset * 4).bindMemory(to: Float.self, capacity: count),
            count: count
        ))
    }

    // MARK: - Step

    /// Incremental step matching QwenCPUModel.step semantics.
    public func step(_ tokens: [Int], state: QwenCPUModel.DecodeState) throws -> [Float] {
        try step(tokens, state: state, shouldCancel: { false })
    }

    /// Cancellable Metal step. Checks happen only while no command buffer is
    /// in flight; a cancelled partial token is abandoned with its DecodeState.
    /// Cancellation exits by throwing, so StepMetrics is published through the
    /// same `defer` as any other failure with `completedWithoutThrow == false`.
    public func step(
        _ tokens: [Int],
        state: QwenCPUModel.DecodeState,
        shouldCancel: () -> Bool
    ) throws -> [Float] {
        let counters = StepCounters()
        if case .dispatchBoundaryCounters = phaseGpuSplitSupport {
            counters.phaseGpuSeconds = [:]
        }
        let wallStart = ProcessInfo.processInfo.systemUptime
        activeStepCounters = counters
        let stepSignpostID = OSSignpostID(log: Self.signpostLog)
        os_signpost(.begin, log: Self.signpostLog, name: "step",
                    signpostID: stepSignpostID, "tokens=%d", tokens.count)
        counters.signpostTally.stepIntervals += 1
        engine.computeDispatchObserver = {
            counters.computeDispatchesEncoded += 1
            let bucket = counters.currentPhase ?? .other
            counters.phaseDispatches[bucket, default: 0] += 1
            if counters.bufferOpen {
                counters.bufferPhaseDispatches[bucket, default: 0] += 1
            }
        }
        defer {
            engine.computeDispatchObserver = nil
            activeStepCounters = nil
            os_signpost(.end, log: Self.signpostLog, name: "step",
                        signpostID: stepSignpostID)
            lastSignpostTally = counters.signpostTally
            lastStepMetrics = StepMetrics(
                tokensProcessed: counters.tokensProcessed,
                logitProjections: counters.logitProjections,
                commandBuffersCommitted: counters.commandBuffersCommitted,
                blockingWaits: counters.blockingWaits,
                blockingWaitSeconds: counters.blockingWaitSeconds,
                computeDispatchesEncoded: counters.computeDispatchesEncoded,
                stepWallSeconds: max(0, ProcessInfo.processInfo.systemUptime - wallStart),
                commandBufferErrors: counters.commandBufferErrors,
                gpuExecutionSeconds: counters.gpuExecutionSeconds,
                gpuTimedCommandBuffers: counters.gpuTimedCommandBuffers,
                gpuUntimedCommandBuffers: counters.gpuUntimedCommandBuffers,
                completedWithoutThrow: counters.completedWithoutThrow,
                commandBufferTimeline: counters.timeline,
                phaseDispatchesEncoded: counters.phaseDispatches,
                phaseEncodeSeconds: counters.phaseEncodeSeconds,
                phaseGpuSeconds: counters.phaseGpuSeconds,
                cpuGap: CPUGapBreakdown(
                    embeddingSeconds: counters.gapEmbeddingSeconds,
                    embeddingLookups: counters.gapEmbeddingLookups,
                    routerSeconds: counters.gapRouterSeconds,
                    expertFetchSeconds: counters.gapExpertFetchSeconds,
                    expertFetchHits: counters.gapExpertFetchHits,
                    expertFetchMisses: counters.gapExpertFetchMisses,
                    kvMirrorSeconds: counters.gapKVMirrorSeconds,
                    commandBufferSetupSeconds: counters.gapCommandBufferSetupSeconds,
                    commitSeconds: counters.gapCommitSeconds,
                    logitsReadbackSeconds: counters.gapLogitsReadbackSeconds
                )
            )
        }

        try ContextWindow(maximumTokens: config.maxPositionEmbeddings).validateStep(
            processedTokens: state.position, incomingTokens: tokens.count
        )

        var logits: [Float] = []
        // S1b: multi-token prompt calls take the layer-major chunked schedule
        // when enabled and the fast path plus chunk scratch are available;
        // everything else keeps the token-major loop (single-token decode
        // steps always do). Cancellation is honoured between chunks and
        // around each token step.
        var chunkCapacity = 0
        if tokens.count > 1, sBuf != nil, hBuf != nil,
           case .layerMajor(let chunkTokens) = prefillMode {
            let wanted = min(max(1, chunkTokens), tokens.count)
            if ensurePrefillCapacity(wanted) { chunkCapacity = wanted }
        }
        if chunkCapacity > 0 {
            logits = try prefillLayerMajor(
                tokens, state: state, chunkCapacity: chunkCapacity, shouldCancel: shouldCancel
            )
        } else {
            for (index, t) in tokens.enumerated() {
                try checkGenerationCancellation(shouldCancel)
                // S1a LM-head elision: a multi-token call consumes only the
                // final position's logits, so intermediate vocabulary
                // projections are unnecessary.
                let projectLogits = index == tokens.count - 1
                logits = try stepOne(
                    t, state: state, projectLogits: projectLogits, shouldCancel: shouldCancel
                )
                state.position += 1
                counters.tokensProcessed += 1
                try checkGenerationCancellation(shouldCancel)
            }
        }
        counters.completedWithoutThrow = true
        return logits
    }

    private func stepOne(
        _ token: Int,
        state: QwenCPUModel.DecodeState,
        projectLogits: Bool,
        shouldCancel: () -> Bool
    ) throws -> [Float] {
        if sBuf != nil {
            return try stepOneFast(
                token, state: state, projectLogits: projectLogits, shouldCancel: shouldCancel
            )
        }
        let cfg = config
        let D = cfg.hiddenSize
        var h = try embeddingRow(token)
        precondition(h.count == D)

        for li in 0..<cfg.numHiddenLayers {
            try checkGenerationCancellation(shouldCancel)
            let layer = layers[li]
            var x = h
            QwenCPUModel.rmsNorm(&x, rows: 1, dim: D, weight: layer.inputNorm, eps: Float(cfg.rmsNormEps))

            let r: [Float]
            if let delta = layer.delta {
                r = try deltaForward(x, w: delta, layerIndex: li, state: state)
            } else {
                r = try attnForward(x, w: layer.attn!, layerIndex: li, state: state)
            }
            for i in 0..<D { h[i] += r[i] }

            var x2 = h
            QwenCPUModel.rmsNorm(&x2, rows: 1, dim: D, weight: layer.postAttnNorm, eps: Float(cfg.rmsNormEps))
            let m = try moeForward(x2, w: layer.moe, layerIndex: li)
            for i in 0..<D { h[i] += m[i] }
        }

        try checkGenerationCancellation(shouldCancel)
        guard projectLogits else { return [] }
        QwenCPUModel.rmsNorm(&h, rows: 1, dim: D, weight: finalNorm, eps: Float(cfg.rmsNormEps))
        loadX(h)
        try runPhase(.lmHead) { enc in
            try engine.encodeGemv(enc, lmHead, x: xBuf, y: logitsBuf, yOff: 0)
        }
        activeStepCounters?.logitProjections += 1
        return readLogits()
    }

    // MARK: - Attention (GQA decode, KV on CPU)

    private func attnForward(_ x: [Float], w: AttnGPU, layerIndex: Int, state: QwenCPUModel.DecodeState) throws -> [Float] {
        let cfg = config
        let H = cfg.numAttentionHeads, KVH = cfg.numKeyValueHeads, hd = cfg.headDim
        let eps = Float(cfg.rmsNormEps)
        let past = state.position
        let qOutDim = H * hd * 2
        let kvDim = KVH * hd

        loadX(x)
        try runPhase(.attention) { enc in
            try engine.encodeGemv(enc, w.qProj, x: xBuf, y: yBuf, yOff: 0)
            try engine.encodeGemv(enc, w.kProj, x: xBuf, y: yBuf, yOff: qOutDim)
            try engine.encodeGemv(enc, w.vProj, x: xBuf, y: yBuf, yOff: qOutDim + kvDim)
        }
        let qOut = readY(0, qOutDim)
        var k = readY(qOutDim, kvDim)
        var v = readY(qOutDim + kvDim, kvDim)

        var q = [Float](repeating: 0, count: H * hd)
        var gate = [Float](repeating: 0, count: H * hd)
        for head in 0..<H {
            for i in 0..<hd {
                q[head * hd + i] = qOut[head * 2 * hd + i]
                gate[head * hd + i] = qOut[head * 2 * hd + hd + i]
            }
        }
        QwenCPUModel.rmsNorm(&q, rows: H, dim: hd, weight: w.qNorm, eps: eps)
        QwenCPUModel.rmsNorm(&k, rows: KVH, dim: hd, weight: w.kNorm, eps: eps)
        applyRope(&q, heads: H, position: past)
        applyRope(&k, heads: KVH, position: past)

        var cache = state.kv[layerIndex] ?? (k: [], v: [])
        gapScope(\.gapKVMirrorSeconds) {
            cache.k.append(contentsOf: k)
            cache.v.append(contentsOf: v)
            state.kv[layerIndex] = cache
        }
        let kAll = cache.k, vAll = cache.v
        let kvLen = past + 1

        let scale = 1 / Float(hd).squareRoot()
        var attnOut = [Float](repeating: 0, count: H * hd)
        let group = H / KVH
        var scores = [Float](repeating: 0, count: kvLen)
        for head in 0..<H {
            let kvHead = head / group
            for sj in 0..<kvLen {
                var dot: Float = 0
                for i in 0..<hd { dot += q[head * hd + i] * kAll[(sj * KVH + kvHead) * hd + i] }
                scores[sj] = dot * scale
            }
            QwenCPUModel.softmaxRow(&scores, base: 0, count: kvLen)
            for sj in 0..<kvLen {
                let p = scores[sj]
                let vBase = (sj * KVH + kvHead) * hd
                for i in 0..<hd { attnOut[head * hd + i] += p * vAll[vBase + i] }
            }
        }
        for i in 0..<attnOut.count { attnOut[i] *= QwenCPUModel.sigmoid(gate[i]) }

        loadX(attnOut)
        try runPhase(.attention) { enc in
            try engine.encodeGemv(enc, w.oProj, x: xBuf, y: yBuf, yOff: 0)
        }
        return readY(0, cfg.hiddenSize)
    }

    private func applyRope(_ x: inout [Float], heads: Int, position: Int) {
        let hd = config.headDim
        let rot = config.rotaryDims
        let half = rot / 2
        for head in 0..<heads {
            let base = head * hd
            for j in 0..<half {
                let invFreq = powf(Float(config.ropeTheta), -Float(2 * j) / Float(rot))
                let angle = Float(position) * invFreq
                let c = cosf(angle), sn = sinf(angle)
                let a = x[base + j]
                let b = x[base + half + j]
                x[base + j] = a * c - b * sn
                x[base + half + j] = b * c + a * sn
            }
        }
    }

    // MARK: - Gated DeltaNet (recurrence on CPU, projections on GPU)

    private func deltaForward(_ x: [Float], w: DeltaGPU, layerIndex: Int, state: QwenCPUModel.DecodeState) throws -> [Float] {
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

        loadX(x)
        switch cfg.deltaLayout {
        case .fusedInterleaved:
            let qkvzDim = 2 * keyDim + 2 * valueDim
            try runPhase(.delta) { enc in
                try engine.encodeGemv(enc, w.qkvz!, x: xBuf, y: yBuf, yOff: 0)
                try engine.encodeGemv(enc, w.ba!, x: xBuf, y: yBuf, yOff: qkvzDim)
            }
            let qkvz = readY(0, qkvzDim)
            let ba = readY(qkvzDim, 2 * nv)
            let chunk = 2 * dk + 2 * rep * dv
            mixedQKV = [Float](repeating: 0, count: convDim)
            z = [Float](repeating: 0, count: valueDim)
            bArr = [Float](repeating: 0, count: nv)
            aArr = [Float](repeating: 0, count: nv)
            for hk in 0..<nk {
                let src = hk * chunk
                for i in 0..<dk {
                    mixedQKV[hk * dk + i] = qkvz[src + i]
                    mixedQKV[keyDim + hk * dk + i] = qkvz[src + dk + i]
                }
                for ri in 0..<rep {
                    let hv = hk * rep + ri
                    for i in 0..<dv {
                        mixedQKV[2 * keyDim + hv * dv + i] = qkvz[src + 2 * dk + ri * dv + i]
                        z[hv * dv + i] = qkvz[src + 2 * dk + rep * dv + ri * dv + i]
                    }
                }
                for ri in 0..<rep {
                    bArr[hk * rep + ri] = ba[hk * 2 * rep + ri]
                    aArr[hk * rep + ri] = ba[hk * 2 * rep + rep + ri]
                }
            }
        case .split:
            try runPhase(.delta) { enc in
                try engine.encodeGemv(enc, w.qkv!, x: xBuf, y: yBuf, yOff: 0)
                try engine.encodeGemv(enc, w.zProj!, x: xBuf, y: yBuf, yOff: convDim)
                try engine.encodeGemv(enc, w.bProj!, x: xBuf, y: yBuf, yOff: convDim + valueDim)
                try engine.encodeGemv(enc, w.aProj!, x: xBuf, y: yBuf, yOff: convDim + valueDim + nv)
            }
            mixedQKV = readY(0, convDim)
            z = readY(convDim, valueDim)
            bArr = readY(convDim + valueDim, nv)
            aArr = readY(convDim + valueDim + nv, nv)
        }

        // Conv step + recurrence: identical math to QwenCPUModel, S = 1.
        let tailRows = K - 1
        var padded = state.convTail[layerIndex] ?? [Float](repeating: 0, count: tailRows * convDim)
        padded.append(contentsOf: mixedQKV)
        var convOut = [Float](repeating: 0, count: convDim)
        for c in 0..<convDim {
            var acc: Float = 0
            for j in 0..<K { acc += w.conv[c * K + j] * padded[j * convDim + c] }
            convOut[c] = QwenCPUModel.silu(acc)
        }
        state.convTail[layerIndex] = Array(padded.suffix(tailRows * convDim))

        var qh = Array(convOut[0..<keyDim])
        var kh = Array(convOut[keyDim..<2 * keyDim])
        let vh = Array(convOut[2 * keyDim..<convDim])
        QwenCPUModel.rmsNorm(&qh, rows: nk, dim: dk, weight: nil, eps: 1e-6)
        QwenCPUModel.rmsNorm(&kh, rows: nk, dim: dk, weight: nil, eps: 1e-6)
        let invScale = 1 / Float(dk).squareRoot()
        for i in 0..<qh.count { qh[i] *= invScale * invScale }
        for i in 0..<kh.count { kh[i] *= invScale }

        var delta0 = state.deltaState[layerIndex] ?? [Float](repeating: 0, count: nv * dv * dk)
        var out = [Float](repeating: 0, count: valueDim)
        for hv in 0..<nv {
            let hk = hv / rep
            let g = expf(-expf(w.aLog[hv]) * QwenCPUModel.softplus(aArr[hv] + w.dtBias[hv]))
            let beta = QwenCPUModel.sigmoid(bArr[hv])
            for dvi in 0..<dv {
                let stBase = (hv * dv + dvi) * dk
                var kvMem: Float = 0
                for dki in 0..<dk {
                    delta0[stBase + dki] *= g
                    kvMem += delta0[stBase + dki] * kh[hk * dk + dki]
                }
                let d = (vh[hv * dv + dvi] - kvMem) * beta
                var yv: Float = 0
                for dki in 0..<dk {
                    delta0[stBase + dki] += kh[hk * dk + dki] * d
                    yv += delta0[stBase + dki] * qh[hk * dk + dki]
                }
                out[hv * dv + dvi] = yv
            }
        }
        state.deltaState[layerIndex] = delta0

        var normed = out
        QwenCPUModel.rmsNorm(&normed, rows: nv, dim: dv, weight: w.norm, eps: Float(cfg.rmsNormEps))
        for i in 0..<normed.count { normed[i] *= QwenCPUModel.silu(z[i]) }

        loadX(normed)
        try runPhase(.delta) { enc in
            try engine.encodeGemv(enc, w.outProj, x: xBuf, y: yBuf, yOff: 0)
        }
        return readY(0, cfg.hiddenSize)
    }

    // MARK: - MoE (router + experts on GPU)

    private func moeForward(_ x: [Float], w: MoEGPU, layerIndex: Int) throws -> [Float] {
        let cfg = config
        let D = cfg.hiddenSize, E = cfg.numExperts, inter = cfg.moeIntermediateSize
        let sharedInter = cfg.sharedExpertIntermediateSize
        let topK = cfg.numExpertsPerTok

        loadX(x)
        try runPhase(.router) { enc in
            try engine.encodeGemv(enc, w.gate, x: xBuf, y: yBuf, yOff: 0)
        }
        var picks: [(Int, Float)] = []
        gapScope(\.gapRouterSeconds) {
            var router = readY(0, E)
            QwenCPUModel.softmaxRow(&router, base: 0, count: E)
            for e in 0..<E {
                let p = router[e]
                if picks.count < topK {
                    picks.append((e, p))
                    picks.sort { $0.1 > $1.1 }
                } else if p > picks[topK - 1].1 {
                    picks[topK - 1] = (e, p)
                    picks.sort { $0.1 > $1.1 }
                }
            }
        }
        routedExpertObserver?(layerIndex, picks.map { $0.0 })

        // Scratch layout in yBuf (floats):
        //   [0, K*inter)                        expert gate outputs
        //   [gBase2, +K*inter)                  expert up outputs
        //   [hBase, +K*inter)                   silu(g)*u
        //   [dBase, +K*D)                       expert down outputs
        //   [shBase, +2*sharedInter+sharedInter+D) shared expert chain
        let K = picks.count
        let gBase2 = K * inter
        let hBase = 2 * K * inter
        let dBase = 3 * K * inter
        let shBase = dBase + K * D

        // Fetch expert buffers up front (qpack path: preads on cache misses).
        var expertBufs: [MTLBuffer] = []
        if let cache = expertCache {
            expertBufs = try fetchExperts(cache, layer: layerIndex, experts: picks.map { $0.0 })
        }

        try runPhase(.moe) { enc in
            for (ki, pick) in picks.enumerated() {
                let e = pick.0
                if let projs = expertProjs {
                    let buf = expertBufs[ki]
                    try engine.encodeGemv(enc, projs.gate.linear(on: buf), x: xBuf, y: yBuf, yOff: ki * inter)
                    try engine.encodeGemv(enc, projs.up.linear(on: buf), x: xBuf, y: yBuf, yOff: gBase2 + ki * inter)
                    try engine.encodeSiluMul(
                        enc, buf: yBuf, count: inter,
                        gOff: ki * inter, uOff: gBase2 + ki * inter, dstOff: hBase + ki * inter
                    )
                    try engine.encodeGemv(
                        enc, projs.down.linear(on: buf),
                        x: yBuf, xOff: hBase + ki * inter, y: yBuf, yOff: dBase + ki * D
                    )
                    continue
                }
                let st = w.stacks!
                try engine.encodeGemv(
                    enc, st.gate.lin, rows: inter,
                    wExtra: e * st.gate.wStride, sExtra: e * st.gate.sStride, bExtra: e * st.gate.sStride,
                    x: xBuf, y: yBuf, yOff: ki * inter
                )
                try engine.encodeGemv(
                    enc, st.up.lin, rows: inter,
                    wExtra: e * st.up.wStride, sExtra: e * st.up.sStride, bExtra: e * st.up.sStride,
                    x: xBuf, y: yBuf, yOff: gBase2 + ki * inter
                )
                try engine.encodeSiluMul(
                    enc, buf: yBuf, count: inter,
                    gOff: ki * inter, uOff: gBase2 + ki * inter, dstOff: hBase + ki * inter
                )
                try engine.encodeGemv(
                    enc, st.down.lin, rows: D,
                    wExtra: e * st.down.wStride, sExtra: e * st.down.sStride, bExtra: e * st.down.sStride,
                    x: yBuf, xOff: hBase + ki * inter, y: yBuf, yOff: dBase + ki * D
                )
            }
            // Shared expert chain.
            try engine.encodeGemv(enc, w.sharedGate, x: xBuf, y: yBuf, yOff: shBase)
            try engine.encodeGemv(enc, w.sharedUp, x: xBuf, y: yBuf, yOff: shBase + sharedInter)
            try engine.encodeSiluMul(
                enc, buf: yBuf, count: sharedInter,
                gOff: shBase, uOff: shBase + sharedInter, dstOff: shBase + 2 * sharedInter
            )
            try engine.encodeGemv(
                enc, w.sharedDown, x: yBuf, xOff: shBase + 2 * sharedInter,
                y: yBuf, yOff: shBase + 3 * sharedInter
            )
        }

        var scoreSum: Float = 0
        for (_, p) in picks { scoreSum += p }
        var out = [Float](repeating: 0, count: D)
        for (ki, pick) in picks.enumerated() {
            let weight = cfg.normTopkProb ? pick.1 / scoreSum : pick.1
            let dOut = readY(dBase + ki * D, D)
            for i in 0..<D { out[i] += weight * dOut[i] }
        }
        var sg: Float = 0
        for i in 0..<D { sg += w.sharedExpertGate[i] * x[i] }
        let sharedScale = QwenCPUModel.sigmoid(sg)
        let shOut = readY(shBase + 3 * sharedInter, D)
        for i in 0..<D { out[i] += sharedScale * shOut[i] }
        return out
    }
}


// MARK: - Fast path: one command buffer per layer in decode (S2 folded the
// attention layers' former two-buffer CPU round trip into one), explicit
// barriers on an untracked scratch buffer so independent dispatches (e.g.
// all expert GEMVs) actually run in parallel.
extension QwenMetalModel {

    /// Where one token's fast-path scratch and hidden row live. The decode
    /// path uses the single legacy slot (sBuf/hBuf at offset zero); a chunked
    /// prefill can give every token its own stride so tokens never collide
    /// inside a shared command buffer. Encoding a token through a slot is
    /// byte-for-byte the legacy schedule when the slot is the legacy one.
    struct TokenSlot {
        /// Scratch buffer holding this token's Regions layout.
        var scratch: MTLBuffer
        /// Float offset added to every Regions offset inside `scratch`.
        var base: Int
        /// Buffer holding this token's hidden-state row.
        var hidden: MTLBuffer
        /// Byte offset of the hidden row inside `hidden`.
        var hiddenByteOffset: Int
    }

    var legacySlot: TokenSlot {
        TokenSlot(scratch: sBuf!, base: 0, hidden: hBuf!, hiddenByteOffset: 0)
    }

    private func setBytesParams<T>(_ enc: MTLComputeCommandEncoder, _ v: inout T, index: Int) {
        enc.setBytes(&v, length: MemoryLayout<T>.stride, index: index)
    }

    private func dispatchRows(_ enc: MTLComputeCommandEncoder, rows: Int) {
        engine.dispatchThreads(
            enc, threads: MTLSize(width: 32, height: rows, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    private func dispatchN(_ enc: MTLComputeCommandEncoder, _ n: Int) {
        engine.dispatchThreads(
            enc, threads: MTLSize(width: n, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, n), height: 1, depth: 1)
        )
    }

    struct NormParams {
        var rows: UInt32
        var dim: UInt32
        var eps: Float
        var hasWeight: UInt32
        var scale: Float
        var off: UInt32
    }

    private func encNormCopy(
        _ enc: MTLComputeCommandEncoder, w: MTLBuffer, dstOff: Int, slot: TokenSlot
    ) throws {
        enc.setComputePipelineState(try engine.pipeline("norm_copy"))
        var p = NormParams(rows: 1, dim: UInt32(config.hiddenSize), eps: Float(config.rmsNormEps),
                           hasWeight: 1, scale: 1, off: UInt32(slot.base + dstOff))
        enc.setBuffer(slot.hidden, offset: slot.hiddenByteOffset, index: 0)
        enc.setBuffer(slot.scratch, offset: 0, index: 1)
        enc.setBuffer(w, offset: 0, index: 2)
        setBytesParams(enc, &p, index: 3)
        dispatchRows(enc, rows: 1)
    }

    private func encRMSNorm(_ enc: MTLComputeCommandEncoder, off: Int, rows: Int, dim: Int,
                            w: MTLBuffer?, scale: Float, eps: Float, slot: TokenSlot) throws {
        enc.setComputePipelineState(try engine.pipeline("rmsnorm_rows"))
        var p = NormParams(rows: UInt32(rows), dim: UInt32(dim), eps: eps,
                           hasWeight: w != nil ? 1 : 0, scale: scale, off: UInt32(slot.base + off))
        enc.setBuffer(slot.scratch, offset: 0, index: 0)
        enc.setBuffer(w ?? slot.scratch, offset: 0, index: 1)
        setBytesParams(enc, &p, index: 2)
        dispatchRows(enc, rows: rows)
    }

    private func barrier(_ enc: MTLComputeCommandEncoder) {
        enc.memoryBarrier(scope: .buffers)
    }

    private func readSlot(_ slot: TokenSlot, _ off: Int, _ n: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: slot.scratch.contents().advanced(by: (slot.base + off) * 4)
                .bindMemory(to: Float.self, capacity: n),
            count: n))
    }

    private func resetGPUState() {
        for fl in fastLayers {
            if let h = fl.hist { memset(h.contents(), 0, h.length) }
            if let st = fl.state { memset(st.contents(), 0, st.length) }
        }
    }

    private func encodePendingMoE(
        _ enc: MTLComputeCommandEncoder, _ pend: PendingMoE, slot: TokenSlot
    ) throws {
        let cfg = config
        let D = cfg.hiddenSize, inter = cfg.moeIntermediateSize
        let shInter = cfg.sharedExpertIntermediateSize
        let moe = layers[pend.stacksLayer].moe
        let fl = fastLayers[pend.stacksLayer]
        let projs = expertProjs

        for (ki, pick) in pend.picks.enumerated() {
            let gLin: GPULinear
            let uLin: GPULinear
            var wE = 0, sE = 0
            if let projs, ki < pend.bufs.count {
                let buf = pend.bufs[ki]
                gLin = projs.gate.linear(on: buf)
                uLin = projs.up.linear(on: buf)
            } else {
                let st = moe.stacks!
                gLin = st.gate.lin; uLin = st.up.lin
                wE = pick.0; sE = pick.0
            }
            try engine.encodeGemv(enc, gLin, rows: inter,
                wExtra: wE * (moe.stacks?.gate.wStride ?? 0), sExtra: sE * (moe.stacks?.gate.sStride ?? 0),
                bExtra: sE * (moe.stacks?.gate.sStride ?? 0),
                x: slot.scratch, xOff: slot.base + reg.xmoe,
                y: slot.scratch, yOff: slot.base + reg.exp + ki * inter)
            try engine.encodeGemv(enc, uLin, rows: inter,
                wExtra: wE * (moe.stacks?.up.wStride ?? 0), sExtra: sE * (moe.stacks?.up.sStride ?? 0),
                bExtra: sE * (moe.stacks?.up.sStride ?? 0),
                x: slot.scratch, xOff: slot.base + reg.xmoe,
                y: slot.scratch, yOff: slot.base + reg.exp + pend.picks.count * inter + ki * inter)
        }
        try engine.encodeGemv(enc, moe.sharedGate, x: slot.scratch, xOff: slot.base + reg.xmoe,
            y: slot.scratch, yOff: slot.base + reg.sh)
        try engine.encodeGemv(enc, moe.sharedUp, x: slot.scratch, xOff: slot.base + reg.xmoe,
            y: slot.scratch, yOff: slot.base + reg.sh + shInter)
        try engine.encodeGemv(enc, fl.sharedGateLin, x: slot.scratch, xOff: slot.base + reg.xmoe,
            y: slot.scratch, yOff: slot.base + reg.shg)
        barrier(enc)
        let K = pend.picks.count
        for ki in 0..<K {
            try engine.encodeSiluMul(enc, buf: slot.scratch, count: inter,
                gOff: slot.base + reg.exp + ki * inter,
                uOff: slot.base + reg.exp + K * inter + ki * inter,
                dstOff: slot.base + reg.exp + 2 * K * inter + ki * inter)
        }
        try engine.encodeSiluMul(enc, buf: slot.scratch, count: shInter,
            gOff: slot.base + reg.sh, uOff: slot.base + reg.sh + shInter,
            dstOff: slot.base + reg.sh + 2 * shInter)
        barrier(enc)
        for (ki, pick) in pend.picks.enumerated() {
            let dLin: GPULinear
            var wE = 0, sE = 0
            if let projs = expertProjs, ki < pend.bufs.count {
                dLin = projs.down.linear(on: pend.bufs[ki])
            } else {
                dLin = moe.stacks!.down.lin
                wE = pick.0; sE = pick.0
            }
            try engine.encodeGemv(enc, dLin, rows: D,
                wExtra: wE * (moe.stacks?.down.wStride ?? 0), sExtra: sE * (moe.stacks?.down.sStride ?? 0),
                bExtra: sE * (moe.stacks?.down.sStride ?? 0),
                x: slot.scratch, xOff: slot.base + reg.exp + 2 * K * inter + ki * inter,
                y: slot.scratch, yOff: slot.base + reg.dexp + ki * D)
        }
        try engine.encodeGemv(enc, moe.sharedDown, x: slot.scratch, xOff: slot.base + reg.sh + 2 * shInter,
            y: slot.scratch, yOff: slot.base + reg.sh + 3 * shInter)
        barrier(enc)

        try encWeightedAccum(enc, weights: pend.weights, count: pend.picks.count, slot: slot)
        barrier(enc)
    }

    /// h[slot] += sum(weights[k] * expertDown[k]) + sigmoid(sharedGate) * sharedDown.
    private func encWeightedAccum(
        _ enc: MTLComputeCommandEncoder, weights: [Float], count: Int, slot: TokenSlot
    ) throws {
        let cfg = config
        let D = cfg.hiddenSize, shInter = cfg.sharedExpertIntermediateSize
        enc.setComputePipelineState(try engine.pipeline("weighted_accum"))
        var ap = SIMD8<UInt32>(UInt32(D), UInt32(count), UInt32(slot.base + reg.dexp),
                               UInt32(slot.base + reg.sh + 3 * shInter),
                               UInt32(slot.base + reg.shg), 0, 0, 0)
        enc.setBuffer(slot.hidden, offset: slot.hiddenByteOffset, index: 0)
        enc.setBuffer(slot.scratch, offset: 0, index: 1)
        setBytesParams(enc, &ap, index: 2)
        weights.withUnsafeBufferPointer {
            enc.setBytes($0.baseAddress!, length: max(1, $0.count) * 4, index: 3)
        }
        dispatchN(enc, D)
    }

    private func routerPicks(slot: TokenSlot) -> ([(Int, Float)], [Float]) {
        let cfg = config
        var router = readSlot(slot, reg.rout, cfg.numExperts)
        QwenCPUModel.softmaxRow(&router, base: 0, count: cfg.numExperts)
        var picks: [(Int, Float)] = []
        for e in 0..<cfg.numExperts {
            let p = router[e]
            if picks.count < cfg.numExpertsPerTok {
                picks.append((e, p)); picks.sort { $0.1 > $1.1 }
            } else if p > picks[cfg.numExpertsPerTok - 1].1 {
                picks[cfg.numExpertsPerTok - 1] = (e, p); picks.sort { $0.1 > $1.1 }
            }
        }
        var sum: Float = 0
        for (_, p) in picks { sum += p }
        let weights = picks.map { cfg.normTopkProb ? $0.1 / sum : $0.1 }
        return (picks, weights)
    }

    /// Attention: input norm + q/k/v projections into the slot's regions.
    private func encAttentionProjections(
        _ enc: MTLComputeCommandEncoder, attn: AttnGPU, fl: FastLayer, slot: TokenSlot
    ) throws {
        try encNormCopy(enc, w: fl.inputNorm, dstOff: reg.x0, slot: slot)
        barrier(enc)
        try engine.encodeGemv(enc, attn.qProj, x: slot.scratch, xOff: slot.base + reg.x0,
            y: slot.scratch, yOff: slot.base + reg.qout)
        try engine.encodeGemv(enc, attn.kProj, x: slot.scratch, xOff: slot.base + reg.x0,
            y: slot.scratch, yOff: slot.base + reg.knew)
        try engine.encodeGemv(enc, attn.vProj, x: slot.scratch, xOff: slot.base + reg.x0,
            y: slot.scratch, yOff: slot.base + reg.vnew)
    }

    /// Attention: output projection and residual add into the slot's hidden row.
    private func encAttentionFinish(
        _ enc: MTLComputeCommandEncoder, attn: AttnGPU, slot: TokenSlot
    ) throws {
        try engine.encodeGemv(enc, attn.oProj, x: slot.scratch, xOff: slot.base + reg.att,
            y: slot.scratch, yOff: slot.base + reg.r)
        barrier(enc)
        try encResidualAdd(enc, slot: slot)
        barrier(enc)
    }

    /// Post-attention norm and router logits projection for one token.
    private func encRouterProbe(
        _ enc: MTLComputeCommandEncoder, moeGate: GPULinear, fl: FastLayer, slot: TokenSlot
    ) throws {
        try encNormCopy(enc, w: fl.postNorm, dstOff: reg.xmoe, slot: slot)
        barrier(enc)
        try engine.encodeGemv(enc, moeGate, x: slot.scratch, xOff: slot.base + reg.xmoe,
            y: slot.scratch, yOff: slot.base + reg.rout)
    }

    /// Final norm + vocabulary projection for the slot's hidden row.
    private func encFinalLMHead(_ enc: MTLComputeCommandEncoder, slot: TokenSlot) throws {
        enc.setComputePipelineState(try engine.pipeline("norm_copy"))
        var np = NormParams(rows: 1, dim: UInt32(config.hiddenSize), eps: Float(config.rmsNormEps),
                            hasWeight: 1, scale: 1, off: UInt32(slot.base + reg.x0))
        enc.setBuffer(slot.hidden, offset: slot.hiddenByteOffset, index: 0)
        enc.setBuffer(slot.scratch, offset: 0, index: 1)
        enc.setBuffer(finalNormBuf!, offset: 0, index: 2)
        setBytesParams(enc, &np, index: 3)
        dispatchRows(enc, rows: 1)
        barrier(enc)
        try engine.encodeGemv(enc, lmHead, x: slot.scratch, xOff: slot.base + reg.x0,
            y: logitsBuf, yOff: 0)
    }

    func stepOneFast(
        _ token: Int,
        state: QwenCPUModel.DecodeState,
        projectLogits: Bool,
        shouldCancel: () -> Bool = { false }
    ) throws -> [Float] {
        let cfg = config
        let D = cfg.hiddenSize
        if boundStateID != ObjectIdentifier(state) || state.position == 0 {
            boundStateID = ObjectIdentifier(state)
            if state.position == 0 { resetGPUState() }
        }

        let h0 = try embeddingRow(token)
        h0.withUnsafeBufferPointer {
            hBuf!.contents().copyMemory(from: $0.baseAddress!, byteCount: D * 4)
        }

        let slot = legacySlot
        var pending: PendingMoE? = nil

        for li in 0..<cfg.numHiddenLayers {
            try checkGenerationCancellation(shouldCancel)
            let L = layers[li]
            let fl = fastLayers[li]

            if let delta = L.delta {
                let (cb, enc) = beginStepCommandBuffer()
                if let p = pending {
                    try phase(.moe) { try encodePendingMoE(enc, p, slot: slot) }
                    pending = nil
                }
                try phase(.delta) {
                    try encDeltaCore(enc, delta: delta, fl: fl, slot: slot)
                }
                try phase(.router) {
                    try encRouterProbe(enc, moeGate: L.moe.gate, fl: fl, slot: slot)
                }
                enc.endEncoding()
                commitAndWait(cb)
                try checkGenerationCancellation(shouldCancel)
            } else {
                // S2: the whole attention layer in one command buffer — no
                // CPU round trip. Projections, q prep + KV append into the
                // GPU-resident cache, causal attention over it, out-proj +
                // residual, then the router probe.
                let attn = L.attn!
                try ensureKVCapacity(li, rows: state.position + 1)
                let (cb, enc) = beginStepCommandBuffer()
                if let p = pending {
                    try phase(.moe) { try encodePendingMoE(enc, p, slot: slot) }
                    pending = nil
                }
                try phase(.attention) {
                    try encAttentionProjections(enc, attn: attn, fl: fl, slot: slot)
                    barrier(enc)
                    try encAttentionPrep(enc, layer: li, position: state.position, slot: slot)
                    barrier(enc)
                    try encAttentionAttend(enc, layer: li, kvLen: state.position + 1, slot: slot)
                    barrier(enc)
                    try encAttentionFinish(enc, attn: attn, slot: slot)
                }
                try phase(.router) {
                    try encRouterProbe(enc, moeGate: L.moe.gate, fl: fl, slot: slot)
                }
                enc.endEncoding()
                commitAndWait(cb)
                gapScope(\.gapKVMirrorSeconds) {
                    appendKVMirror(state, layer: li, position: state.position, count: 1)
                }
                try checkGenerationCancellation(shouldCancel)
            }

            let (picks, weights) = gapScope(\.gapRouterSeconds) { routerPicks(slot: slot) }
            routedExpertObserver?(li, picks.map { $0.0 })
            var bufs: [MTLBuffer] = []
            if let cache = expertCache {
                try checkGenerationCancellation(shouldCancel)
                bufs = try fetchExperts(cache, layer: li, experts: picks.map { $0.0 })
            }
            pending = PendingMoE(bufs: bufs, weights: weights, stacksLayer: li, picks: picks)
        }

        // Always apply the last layer's experts. Only the final token in a
        // multi-token call also needs final norm + vocabulary projection.
        try checkGenerationCancellation(shouldCancel)
        let (cb, enc) = beginStepCommandBuffer()
        if let p = pending { try phase(.moe) { try encodePendingMoE(enc, p, slot: slot) } }
        if projectLogits {
            try phase(.lmHead) {
                try encFinalLMHead(enc, slot: slot)
            }
            activeStepCounters?.logitProjections += 1
        }
        enc.endEncoding()
        commitAndWait(cb)
        try checkGenerationCancellation(shouldCancel)

        guard projectLogits else { return [] }
        return readLogits()
    }

    /// DeltaNet mixer for one token, entirely on GPU: projections, conv step,
    /// gated recurrence, gated norm, output projection, residual add. Encoded
    /// through the stage helpers below in exactly the legacy order; the
    /// chunked prefill reuses the stages but batches the projection GEMVs
    /// across tokens (S1b token batching).
    private func encDeltaCore(
        _ enc: MTLComputeCommandEncoder, delta: DeltaGPU, fl: FastLayer, slot: TokenSlot
    ) throws {
        try encNormCopy(enc, w: fl.inputNorm, dstOff: reg.x0, slot: slot)
        barrier(enc)
        for proj in deltaInProjections(delta) {
            try engine.encodeGemv(enc, proj.lin, x: slot.scratch, xOff: slot.base + reg.x0,
                y: slot.scratch, yOff: slot.base + proj.dst)
        }
        barrier(enc)
        if config.deltaLayout == .fusedInterleaved {
            try encDeltaDeinterleave(enc, slot: slot)
            barrier(enc)
        }
        try encDeltaRecurrence(enc, fl: fl, slot: slot)
        try engine.encodeGemv(enc, delta.outProj, x: slot.scratch, xOff: slot.base + reg.dn,
            y: slot.scratch, yOff: slot.base + reg.r)
        barrier(enc)
        try encResidualAdd(enc, slot: slot)
        barrier(enc)
    }

    /// The DeltaNet input projections as (linear, destination region) pairs
    /// in the layout's canonical encode order; the decode core and the
    /// batched chunk pass dispatch exactly these.
    private func deltaInProjections(_ delta: DeltaGPU) -> [(lin: GPULinear, dst: Int)] {
        switch config.deltaLayout {
        case .split:
            return [(delta.qkv!, reg.qkv), (delta.zProj!, reg.z),
                    (delta.bProj!, reg.b), (delta.aProj!, reg.a)]
        case .fusedInterleaved:
            return [(delta.qkvz!, reg.qkvzStage), (delta.ba!, reg.baStage)]
        }
    }

    /// Splits the fused qkvz/ba staging rows into the flat qkv/z/b/a regions
    /// (fusedInterleaved layout only).
    private func encDeltaDeinterleave(_ enc: MTLComputeCommandEncoder, slot: TokenSlot) throws {
        let cfg = config
        let nk = cfg.linearNumKeyHeads, nv = cfg.linearNumValueHeads
        let dk = cfg.linearKeyHeadDim, dv = cfg.linearValueHeadDim
        enc.setComputePipelineState(try engine.pipeline("deinterleave_qkvz"))
        struct DeintParams {
            var nk: UInt32; var dk: UInt32; var rep: UInt32; var dv: UInt32
            var keyDim: UInt32; var srcOff: UInt32; var baOff: UInt32
            var qkvOff: UInt32; var zOff: UInt32; var bOff: UInt32; var aOff: UInt32
        }
        var dip = DeintParams(
            nk: UInt32(nk), dk: UInt32(dk), rep: UInt32(nv / nk), dv: UInt32(dv),
            keyDim: UInt32(cfg.keyDim),
            srcOff: UInt32(slot.base + reg.qkvzStage), baOff: UInt32(slot.base + reg.baStage),
            qkvOff: UInt32(slot.base + reg.qkv), zOff: UInt32(slot.base + reg.z),
            bOff: UInt32(slot.base + reg.b), aOff: UInt32(slot.base + reg.a)
        )
        enc.setBuffer(slot.scratch, offset: 0, index: 0)
        setBytesParams(enc, &dip, index: 1)
        dispatchN(enc, nk)
    }

    /// Conv step, decay/beta preparation, q/k norms, gated recurrence, and
    /// the gated output norm for one token, ending on a barrier. Reads the
    /// slot's projection regions and advances the layer's shared conv
    /// history and recurrent state, so chunked callers must encode tokens in
    /// ascending order (the decode core encodes exactly one). Assembled
    /// from the stage helpers below; the chunked prefill batches decay/beta
    /// prep, the q/k norms, and the gated norm through their stride twins
    /// and runs each order-sensitive stage as one cross-token scan per
    /// chunk (conv_scan, and the gated scan with T = chunk length; S1b
    /// chunked recurrence).
    private func encDeltaRecurrence(
        _ enc: MTLComputeCommandEncoder, fl: FastLayer, slot: TokenSlot
    ) throws {
        let nv = config.linearNumValueHeads
        let keyDim = config.keyDim
        try encConvStep(enc, fl: fl, slot: slot)
        try encDeltaPre(enc, fl: fl, slot: slot)
        barrier(enc)
        try encDeltaQKNorms(enc, slot: slot)
        barrier(enc)
        try encGatedDeltaScan(
            enc, fl: fl, steps: 1, data: slot.scratch,
            qOff: slot.base + reg.conv, kOff: slot.base + reg.conv + keyDim,
            vOff: slot.base + reg.conv + 2 * keyDim,
            gOff: slot.base + reg.gb, bOff: slot.base + reg.gb + nv,
            yOff: slot.base + reg.dy)
        barrier(enc)
        try encGatedNormMul(
            enc, fl: fl, data: slot.scratch,
            yOff: slot.base + reg.dy, zOff: slot.base + reg.z,
            outOff: slot.base + reg.dn)
        barrier(enc)
    }

    /// Depthwise causal conv + silu on one token's qkv row, advancing the
    /// layer's shared history (order-sensitive across tokens).
    private func encConvStep(
        _ enc: MTLComputeCommandEncoder, fl: FastLayer, slot: TokenSlot
    ) throws {
        let cfg = config
        enc.setComputePipelineState(try engine.pipeline("conv_step"))
        var cp = SIMD4<UInt32>(UInt32(cfg.convDim), UInt32(cfg.linearConvKernelDim),
                               UInt32(slot.base + reg.qkv), UInt32(slot.base + reg.conv))
        enc.setBuffer(slot.scratch, offset: 0, index: 0)
        enc.setBuffer(fl.hist!, offset: 0, index: 1)
        enc.setBuffer(fl.convW!, offset: 0, index: 2)
        setBytesParams(enc, &cp, index: 3)
        dispatchN(enc, cfg.convDim)
    }

    /// g = exp(-exp(A_log) * softplus(a + dt_bias)) and beta = sigmoid(b)
    /// into the slot's gb region.
    private func encDeltaPre(
        _ enc: MTLComputeCommandEncoder, fl: FastLayer, slot: TokenSlot
    ) throws {
        let nv = config.linearNumValueHeads
        enc.setComputePipelineState(try engine.pipeline("delta_pre"))
        var dp = SIMD4<UInt32>(UInt32(nv), UInt32(slot.base + reg.a),
                               UInt32(slot.base + reg.b), UInt32(slot.base + reg.gb))
        enc.setBuffer(slot.scratch, offset: 0, index: 0)
        enc.setBuffer(fl.aLog!, offset: 0, index: 1)
        enc.setBuffer(fl.dtBias!, offset: 0, index: 2)
        setBytesParams(enc, &dp, index: 3)
        dispatchN(enc, nv)
    }

    /// Weightless q/k RMSNorms (scaled 1/dk and 1/sqrt(dk)) in place on the
    /// slot's conv rows.
    private func encDeltaQKNorms(_ enc: MTLComputeCommandEncoder, slot: TokenSlot) throws {
        let cfg = config
        let nk = cfg.linearNumKeyHeads, dk = cfg.linearKeyHeadDim
        let invScale = 1 / Float(dk).squareRoot()
        try encRMSNorm(enc, off: reg.conv, rows: nk, dim: dk, w: nil,
                       scale: invScale * invScale, eps: 1e-6, slot: slot)
        try encRMSNorm(enc, off: reg.conv + cfg.keyDim, rows: nk, dim: dk, w: nil,
                       scale: invScale, eps: 1e-6, slot: slot)
    }

    /// One gated_delta_step dispatch scanning `steps` timesteps from the
    /// q/k/v/g/beta float offsets in `data` (contiguous [steps, row] blocks
    /// when steps > 1), reading and writing the layer's recurrent state.
    private func encGatedDeltaScan(
        _ enc: MTLComputeCommandEncoder, fl: FastLayer, steps: Int, data: MTLBuffer,
        qOff: Int, kOff: Int, vOff: Int, gOff: Int, bOff: Int, yOff: Int
    ) throws {
        let cfg = config
        let nk = cfg.linearNumKeyHeads, nv = cfg.linearNumValueHeads
        let dk = cfg.linearKeyHeadDim, dv = cfg.linearValueHeadDim
        enc.setComputePipelineState(try engine.pipeline("gated_delta_step"))
        var delp = MetalEngine.DeltaParams(
            T: UInt32(steps), Hk: UInt32(nk), Hv: UInt32(nv), Dk: UInt32(dk), Dv: UInt32(dv))
        enc.setBuffer(data, offset: qOff * 4, index: 0)
        enc.setBuffer(data, offset: kOff * 4, index: 1)
        enc.setBuffer(data, offset: vOff * 4, index: 2)
        enc.setBuffer(data, offset: gOff * 4, index: 3)
        enc.setBuffer(data, offset: bOff * 4, index: 4)
        enc.setBuffer(fl.state!, offset: 0, index: 5)
        enc.setBuffer(data, offset: yOff * 4, index: 6)
        enc.setBuffer(fl.state!, offset: 0, index: 7)
        setBytesParams(enc, &delp, index: 8)
        engine.dispatchThreads(
            enc, threads: MTLSize(width: 32, height: dv, depth: nv),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
        )
    }

    /// Gated RMSNorm * silu(z) per v-head, y rows read at `yOff` in `data`.
    private func encGatedNormMul(
        _ enc: MTLComputeCommandEncoder, fl: FastLayer, data: MTLBuffer,
        yOff: Int, zOff: Int, outOff: Int
    ) throws {
        let cfg = config
        let nv = cfg.linearNumValueHeads, dv = cfg.linearValueHeadDim
        enc.setComputePipelineState(try engine.pipeline("gated_norm_mul"))
        struct GNP { var nv: UInt32; var dv: UInt32; var eps: Float; var yOff: UInt32; var zOff: UInt32; var outOff: UInt32 }
        var gp = GNP(nv: UInt32(nv), dv: UInt32(dv), eps: Float(cfg.rmsNormEps),
                     yOff: UInt32(yOff), zOff: UInt32(zOff), outOff: UInt32(outOff))
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(fl.deltaNormW!, offset: 0, index: 1)
        setBytesParams(enc, &gp, index: 2)
        dispatchRows(enc, rows: nv)
    }

    /// h[slot] += the slot's reg.r row (a mixer's output-projection result).
    private func encResidualAdd(_ enc: MTLComputeCommandEncoder, slot: TokenSlot) throws {
        let D = config.hiddenSize
        enc.setComputePipelineState(try engine.pipeline("add_inplace"))
        var p = SIMD2<UInt32>(UInt32(D), UInt32(slot.base + reg.r))
        enc.setBuffer(slot.hidden, offset: slot.hiddenByteOffset, index: 0)
        enc.setBuffer(slot.scratch, offset: 0, index: 1)
        setBytesParams(enc, &p, index: 2)
        dispatchN(enc, D)
    }

    // MARK: - S2 GPU-resident KV cache

    /// GPU KV cache bookkeeping failed; the fast path has no CPU fallback
    /// for attention, so the step surfaces the allocation failure.
    public enum RuntimeError: Swift.Error {
        case kvCacheAllocationFailed(layer: Int, rows: Int)
    }

    private var kvRowFloats: Int { config.numKeyValueHeads * config.headDim }

    /// Grows layer `li`'s GPU KV cache to hold at least `rows` positions
    /// (doubling; written rows copied across). Only called between command
    /// buffers, when every prior buffer has been waited on.
    func ensureKVCapacity(_ li: Int, rows: Int) throws {
        let current = fastLayers[li].kvCapacity
        if current >= rows, fastLayers[li].kCache != nil { return }
        let newCapacity = max(rows, max(256, 2 * current))
        let bytes = newCapacity * kvRowFloats * 4
        let opts: MTLResourceOptions = [.storageModeShared, .hazardTrackingModeUntracked]
        guard let k = engine.device.makeBuffer(length: bytes, options: opts),
              let v = engine.device.makeBuffer(length: bytes, options: opts)
        else { throw RuntimeError.kvCacheAllocationFailed(layer: li, rows: rows) }
        if let oldK = fastLayers[li].kCache, let oldV = fastLayers[li].vCache, current > 0 {
            memcpy(k.contents(), oldK.contents(), current * kvRowFloats * 4)
            memcpy(v.contents(), oldV.contents(), current * kvRowFloats * 4)
        }
        fastLayers[li].kCache = k
        fastLayers[li].vCache = v
        fastLayers[li].kvCapacity = newCapacity
    }

    /// Appends rows [position, position + count) of layer `li`'s GPU KV
    /// cache to the CPU-side mirror in state.kv. The GPU rows are the
    /// source of truth for the attention math; the mirror keeps state.kv the
    /// observable KV state (parity assertions and persistence read it) at
    /// the cost of one memcpy of the new rows per layer. Appended in place
    /// through the dictionary's modify accessor: copying the entry out,
    /// appending, and storing it back duplicates the whole history every
    /// step (S3c measured that at 1.5 ms/step by 567 rows, growing linearly).
    func appendKVMirror(
        _ state: QwenCPUModel.DecodeState, layer li: Int, position: Int, count: Int
    ) {
        guard let k = fastLayers[li].kCache, let v = fastLayers[li].vCache else { return }
        let n = count * kvRowFloats
        let byteOff = position * kvRowFloats * 4
        state.kv[li, default: (k: [], v: [])].k.append(contentsOf: UnsafeBufferPointer(
            start: k.contents().advanced(by: byteOff).bindMemory(to: Float.self, capacity: n),
            count: n))
        state.kv[li, default: (k: [], v: [])].v.append(contentsOf: UnsafeBufferPointer(
            start: v.contents().advanced(by: byteOff).bindMemory(to: Float.self, capacity: n),
            count: n))
    }

    /// Test/introspection hook: layer `li`'s GPU-resident KV rows for
    /// positions [0, count), or nil when the layer keeps no cache (DeltaNet
    /// layers, or attention before its first step). Stays internal.
    func attentionKVCacheRows(layer li: Int, count: Int) -> (k: [Float], v: [Float])? {
        guard let k = fastLayers[li].kCache, let v = fastLayers[li].vCache,
              fastLayers[li].kvCapacity >= count else { return nil }
        let n = count * kvRowFloats
        return (
            k: Array(UnsafeBufferPointer(
                start: k.contents().bindMemory(to: Float.self, capacity: n), count: n)),
            v: Array(UnsafeBufferPointer(
                start: v.contents().bindMemory(to: Float.self, capacity: n), count: n))
        )
    }

    /// Attention core, stage 1 of 2 on GPU: query prep (extract + RMSNorm +
    /// RoPE into reg.qprep) and the KV append for `position` (K normed +
    /// roped, V verbatim). Callers barrier between the q/k/v projections and
    /// this, and between this and the attend stage — batched prefill must
    /// append every token's rows before any token attends.
    func encAttentionPrep(
        _ enc: MTLComputeCommandEncoder, layer li: Int, position: Int, slot: TokenSlot
    ) throws {
        let cfg = config
        let fl = fastLayers[li] // fresh copy: ensureKVCapacity may have swapped buffers
        try engine.encodeAttnQPrep(
            enc, data: slot.scratch, weight: fl.qNormW!,
            heads: cfg.numAttentionHeads, headDim: cfg.headDim,
            rotaryDims: cfg.rotaryDims, eps: Float(cfg.rmsNormEps),
            ropeTheta: Float(cfg.ropeTheta), position: position,
            srcOff: slot.base + reg.qout, dstOff: slot.base + reg.qprep
        )
        try engine.encodeAttnKVAppend(
            enc, data: slot.scratch, weight: fl.kNormW!,
            kCache: fl.kCache!, vCache: fl.vCache!,
            kvHeads: cfg.numKeyValueHeads, headDim: cfg.headDim,
            rotaryDims: cfg.rotaryDims, eps: Float(cfg.rmsNormEps),
            ropeTheta: Float(cfg.ropeTheta), position: position,
            kSrcOff: slot.base + reg.knew, vSrcOff: slot.base + reg.vnew
        )
    }

    /// Attention core, stage 2 of 2: gated causal softmax attention over
    /// cache rows [0, kvLen) into the slot's att region.
    func encAttentionAttend(
        _ enc: MTLComputeCommandEncoder, layer li: Int, kvLen: Int, slot: TokenSlot
    ) throws {
        let cfg = config
        let fl = fastLayers[li] // fresh copy: ensureKVCapacity may have swapped buffers
        try engine.encodeAttnDecode(
            enc, data: slot.scratch, kCache: fl.kCache!, vCache: fl.vCache!,
            heads: cfg.numAttentionHeads, kvHeads: cfg.numKeyValueHeads,
            headDim: cfg.headDim, kvLen: kvLen,
            qOff: slot.base + reg.qprep, qoutOff: slot.base + reg.qout,
            outOff: slot.base + reg.att
        )
    }

}

// MARK: - S1b layer-major chunked prefill: inside a chunk, sweep each layer
// across every token before moving on, so a layer's expert union is fetched
// once per chunk and one decode-shaped buffer sequence covers the whole chunk
// instead of every token. Independent per-token work over shared weights
// encodes as token-batched dispatches: projection GEMVs (one per projection
// per chunk, one per (expert, projection) for routed experts) and the glue
// kernels (norms, deinterleave, silu, residual adds, weighted accumulation,
// attention q prep / KV append) via their batch twins. Every batched row is
// the single-token kernel's arithmetic verbatim; the order-sensitive
// recurrent stages keep ascending token order inside cross-token scans (the
// conv history and the gated recurrence each advance in one T=chunk
// in-dispatch scan), so numerics match the token-major schedule exactly. S2 batches the attention core across the
// chunk's tokens on GPU, including the causal attend: its batch twin derives
// each token's kvLen from the token grid, and every KV append lands behind a
// barrier before any token attends.
extension QwenMetalModel {

    /// One layer's deferred MoE for a whole chunk: per-token picks/weights in
    /// the legacy order, plus the layer's ascending expert union with each
    /// expert's resident buffer fetched exactly once (qpack mode; raw
    /// checkpoints address stacked tensors by stride and keep this empty).
    struct PendingChunkMoE {
        var layer: Int
        var union: [Int]
        var buffers: [Int: MTLBuffer]
        var perToken: [(picks: [(Int, Float)], weights: [Float])]
    }

    /// Where the chunked DeltaNet recurrence stages its contiguous
    /// [chunk, row] blocks (float offsets inside prefillScratchBuf, after
    /// every token slot stride): the normed q/k/v conv rows, g/beta, and
    /// the y output — the exact layout gated_delta_step's T-step scan
    /// expects.
    struct PrefillStage {
        var q = 0, k = 0, v = 0, g = 0, b = 0, y = 0
    }

    /// Grows the chunk scratch to hold `n` token slots plus the chunked
    /// recurrence staging blocks. Returns false when the device cannot
    /// allocate them (callers fall back to token-major).
    private func ensurePrefillCapacity(_ n: Int) -> Bool {
        if prefillSlotCapacity >= n, prefillScratchBuf != nil, prefillHiddenBuf != nil {
            return true
        }
        let opts: MTLResourceOptions = [.storageModeShared, .hazardTrackingModeUntracked]
        let cfg = config
        var stage = PrefillStage()
        var floats = n * reg.total
        func take(_ rowFloats: Int) -> Int {
            let v = floats
            floats += n * rowFloats
            return v
        }
        stage.q = take(cfg.keyDim)
        stage.k = take(cfg.keyDim)
        stage.v = take(cfg.valueDim)
        stage.g = take(cfg.linearNumValueHeads)
        stage.b = take(cfg.linearNumValueHeads)
        stage.y = take(cfg.valueDim)
        guard let scratch = engine.device.makeBuffer(length: floats * 4, options: opts),
              let hidden = engine.device.makeBuffer(length: n * cfg.hiddenSize * 4, options: opts)
        else { return false }
        prefillScratchBuf = scratch
        prefillHiddenBuf = hidden
        prefillSlotCapacity = n
        prefillStage = stage
        return true
    }

    private func prefillSlot(_ t: Int) -> TokenSlot {
        TokenSlot(scratch: prefillScratchBuf!, base: t * reg.total,
                  hidden: prefillHiddenBuf!, hiddenByteOffset: t * config.hiddenSize * 4)
    }

    /// Offset table for one token-batched projection: chunk token t reads
    /// `src` and writes `dst` inside its own slot stride.
    private func chunkGemvSlots(_ n: Int, src: Int, dst: Int) -> [(xOff: Int, yOff: Int)] {
        (0..<n).map { t in (xOff: t * reg.total + src, yOff: t * reg.total + dst) }
    }

    // MARK: Chunk-level batch twins for the per-token glue kernels. Every
    // helper encodes one batched dispatch covering all chunk tokens, and a
    // single-token chunk delegates to the legacy per-token encode so the
    // degenerate chunk keeps the decode-shaped stream byte-identical.

    /// Input/post norms for every chunk token as one norm_copy_batch.
    private func encChunkNormCopy(
        _ enc: MTLComputeCommandEncoder, w: MTLBuffer, dstOff: Int, tokens n: Int
    ) throws {
        if n == 1 {
            try encNormCopy(enc, w: w, dstOff: dstOff, slot: prefillSlot(0))
            return
        }
        try engine.encodeNormCopyBatch(
            enc, h: prefillHiddenBuf!, hByteOffset: 0, hStride: config.hiddenSize,
            dst: prefillScratchBuf!, weight: w,
            dim: config.hiddenSize, eps: Float(config.rmsNormEps),
            dstOff: dstOff, dstStride: reg.total, tokens: n)
    }

    /// Residual adds for every chunk token as one add_inplace_batch.
    private func encChunkResidualAdd(_ enc: MTLComputeCommandEncoder, tokens n: Int) throws {
        if n == 1 {
            try encResidualAdd(enc, slot: prefillSlot(0))
            return
        }
        try engine.encodeAddInplaceBatch(
            enc, h: prefillHiddenBuf!, hStride: config.hiddenSize,
            r: prefillScratchBuf!, rOff: reg.r, rStride: reg.total,
            count: config.hiddenSize, tokens: n)
    }

    /// Fused qkvz/ba splits for every chunk token as one
    /// deinterleave_qkvz_batch (fusedInterleaved layout only).
    private func encChunkDeinterleave(_ enc: MTLComputeCommandEncoder, tokens n: Int) throws {
        if n == 1 {
            try encDeltaDeinterleave(enc, slot: prefillSlot(0))
            return
        }
        let cfg = config
        try engine.encodeDeinterleaveBatch(
            enc, data: prefillScratchBuf!,
            nk: cfg.linearNumKeyHeads, dk: cfg.linearKeyHeadDim,
            rep: cfg.linearNumValueHeads / cfg.linearNumKeyHeads,
            dv: cfg.linearValueHeadDim, keyDim: cfg.keyDim,
            srcOff: reg.qkvzStage, baOff: reg.baStage,
            qkvOff: reg.qkv, zOff: reg.z, bOff: reg.b, aOff: reg.a,
            stride: reg.total, tokens: n)
    }

    /// Query prep and KV append for every chunk token: one batched dispatch
    /// each. Token t preps at position basePosition + t and appends into its
    /// own (disjoint) cache row, so batching cannot reorder any write.
    private func encChunkAttentionPrep(
        _ enc: MTLComputeCommandEncoder, layer li: Int, basePosition: Int, tokens n: Int
    ) throws {
        if n == 1 {
            try encAttentionPrep(enc, layer: li, position: basePosition, slot: prefillSlot(0))
            return
        }
        let cfg = config
        let fl = fastLayers[li] // fresh copy: ensureKVCapacity may have swapped buffers
        try engine.encodeAttnQPrepBatch(
            enc, data: prefillScratchBuf!, weight: fl.qNormW!,
            heads: cfg.numAttentionHeads, headDim: cfg.headDim,
            rotaryDims: cfg.rotaryDims, eps: Float(cfg.rmsNormEps),
            ropeTheta: Float(cfg.ropeTheta), basePosition: basePosition,
            srcOff: reg.qout, dstOff: reg.qprep, slotStride: reg.total, tokens: n)
        try engine.encodeAttnKVAppendBatch(
            enc, data: prefillScratchBuf!, weight: fl.kNormW!,
            kCache: fl.kCache!, vCache: fl.vCache!,
            kvHeads: cfg.numKeyValueHeads, headDim: cfg.headDim,
            rotaryDims: cfg.rotaryDims, eps: Float(cfg.rmsNormEps),
            ropeTheta: Float(cfg.ropeTheta), basePosition: basePosition,
            kSrcOff: reg.knew, vSrcOff: reg.vnew, slotStride: reg.total, tokens: n)
    }

    /// Causal attends for every chunk token as one attn_decode_gqa_batch:
    /// token t attends over cache rows [0, basePosition + t + 1), a kvLen
    /// the kernel derives from the token grid. Callers must barrier between
    /// the chunk's KV appends and this encode — the youngest token reads
    /// every appended row.
    private func encChunkAttentionAttend(
        _ enc: MTLComputeCommandEncoder, layer li: Int, basePosition: Int, tokens n: Int
    ) throws {
        if n == 1 {
            try encAttentionAttend(
                enc, layer: li, kvLen: basePosition + 1, slot: prefillSlot(0))
            return
        }
        let cfg = config
        let fl = fastLayers[li] // fresh copy: ensureKVCapacity may have swapped buffers
        try engine.encodeAttnDecodeBatch(
            enc, data: prefillScratchBuf!, kCache: fl.kCache!, vCache: fl.vCache!,
            heads: cfg.numAttentionHeads, kvHeads: cfg.numKeyValueHeads,
            headDim: cfg.headDim, basePosition: basePosition,
            qOff: reg.qprep, qoutOff: reg.qout, outOff: reg.att,
            slotStride: reg.total, tokens: n)
    }

    /// Splits the prompt into chunks of at most `chunkCapacity` tokens and
    /// prefills each chunk layer-major. Only the final chunk projects logits
    /// (the S1a single-LM-head property).
    func prefillLayerMajor(
        _ tokens: [Int], state: QwenCPUModel.DecodeState, chunkCapacity: Int,
        shouldCancel: () -> Bool = { false }
    ) throws -> [Float] {
        var logits: [Float] = []
        var start = 0
        while start < tokens.count {
            // Each chunk is its own group of command buffers; a cancelled
            // prefill is abandoned here between chunks, before the next one.
            try checkGenerationCancellation(shouldCancel)
            let end = min(start + chunkCapacity, tokens.count)
            logits = try prefillChunkLayerMajor(
                Array(tokens[start..<end]), state: state,
                projectLogits: end == tokens.count
            )
            start = end
        }
        return logits
    }

    /// One chunk, layer-major. The buffer sequence per chunk is exactly the
    /// decode per-token shape (one buffer per layer, one tail), each buffer
    /// now carrying every chunk token's work for that layer in per-token
    /// slots. Within a layer, tokens are encoded in ascending order so the
    /// conv history, recurrence state, and KV appends advance exactly as the
    /// token-major path does.
    private func prefillChunkLayerMajor(
        _ chunk: [Int], state: QwenCPUModel.DecodeState, projectLogits: Bool
    ) throws -> [Float] {
        let cfg = config
        let D = cfg.hiddenSize
        let S = chunk.count
        if boundStateID != ObjectIdentifier(state) || state.position == 0 {
            boundStateID = ObjectIdentifier(state)
            if state.position == 0 { resetGPUState() }
        }
        let basePosition = state.position

        for (t, token) in chunk.enumerated() {
            let h0 = try embeddingRow(token)
            h0.withUnsafeBufferPointer {
                prefillHiddenBuf!.contents().advanced(by: t * D * 4)
                    .copyMemory(from: $0.baseAddress!, byteCount: D * 4)
            }
        }

        var pending: PendingChunkMoE?

        for li in 0..<cfg.numHiddenLayers {
            let L = layers[li]
            let fl = fastLayers[li]

            if let delta = L.delta {
                let (cb, enc) = beginStepCommandBuffer()
                if let p = pending {
                    try phase(.moe) { try encodeChunkMoE(enc, p) }
                    pending = nil
                }
                try phase(.delta) {
                    try encChunkDelta(enc, delta: delta, fl: fl, tokens: S)
                }
                try phase(.router) {
                    try encChunkRouterProbe(enc, moeGate: L.moe.gate, fl: fl, tokens: S)
                }
                enc.endEncoding()
                commitAndWait(cb)
            } else {
                // S2 batched attention: one command buffer carries the whole
                // chunk. Every token's q/k/v projections, then every token's
                // KV append (rows land at disjoint absolute positions), one
                // barrier, and only then the per-token causal attention —
                // token t reads rows [0, basePosition + t + 1), which the
                // append stage has fully written. Ascending-token state
                // semantics hold because position order, not encode order,
                // indexes the cache.
                let attn = L.attn!
                try ensureKVCapacity(li, rows: basePosition + S)
                let (cb, enc) = beginStepCommandBuffer()
                if let p = pending {
                    try phase(.moe) { try encodeChunkMoE(enc, p) }
                    pending = nil
                }
                try phase(.attention) {
                    // One batched input norm over every slot, then q/k/v each
                    // as one token-batched dispatch.
                    try encChunkNormCopy(enc, w: fl.inputNorm, dstOff: reg.x0, tokens: S)
                    barrier(enc)
                    for proj in [(attn.qProj, reg.qout), (attn.kProj, reg.knew),
                                 (attn.vProj, reg.vnew)] {
                        try engine.encodeGemvBatch(
                            enc, proj.0, x: prefillScratchBuf!, y: prefillScratchBuf!,
                            slots: chunkGemvSlots(S, src: reg.x0, dst: proj.1))
                    }
                    barrier(enc)
                    try encChunkAttentionPrep(
                        enc, layer: li, basePosition: basePosition, tokens: S)
                    barrier(enc)
                    // One batched causal attend: token t reads cache rows
                    // [0, basePosition + t + 1), a kvLen the kernel derives
                    // from the token grid; the barrier above guarantees
                    // every row the youngest token can see has landed.
                    try encChunkAttentionAttend(
                        enc, layer: li, basePosition: basePosition, tokens: S)
                    barrier(enc)
                    try engine.encodeGemvBatch(
                        enc, attn.oProj, x: prefillScratchBuf!, y: prefillScratchBuf!,
                        slots: chunkGemvSlots(S, src: reg.att, dst: reg.r))
                    barrier(enc)
                    try encChunkResidualAdd(enc, tokens: S)
                    barrier(enc)
                }
                try phase(.router) {
                    try encChunkRouterProbe(enc, moeGate: L.moe.gate, fl: fl, tokens: S)
                }
                enc.endEncoding()
                commitAndWait(cb)
                gapScope(\.gapKVMirrorSeconds) {
                    appendKVMirror(state, layer: li, position: basePosition, count: S)
                }
            }

            // Routing on CPU for every chunk token, then one union fetch for
            // the whole chunk: the schedule consumes exactly what the S1b-a
            // plan predicts from these routes.
            var perToken: [(picks: [(Int, Float)], weights: [Float])] = []
            var unionSet = Set<Int>()
            for t in 0..<S {
                let (picks, weights) = gapScope(\.gapRouterSeconds) {
                    routerPicks(slot: prefillSlot(t))
                }
                routedExpertObserver?(li, picks.map { $0.0 })
                unionSet.formUnion(picks.map { $0.0 })
                perToken.append((picks, weights))
            }
            let union = unionSet.sorted()
            prefillExpertUnionObserver?(li, union)
            var buffers: [Int: MTLBuffer] = [:]
            if let cache = expertCache {
                let bufs = try fetchExperts(cache, layer: li, experts: union)
                for (i, e) in union.enumerated() { buffers[e] = bufs[i] }
            }
            pending = PendingChunkMoE(
                layer: li, union: union, buffers: buffers, perToken: perToken)
        }

        // Tail: flush the last layer's experts for the whole chunk; only the
        // prompt's final token also needs final norm + vocabulary projection.
        let (cb, enc) = beginStepCommandBuffer()
        if let p = pending { try phase(.moe) { try encodeChunkMoE(enc, p) } }
        if projectLogits {
            try phase(.lmHead) {
                try encFinalLMHead(enc, slot: prefillSlot(S - 1))
            }
            activeStepCounters?.logitProjections += 1
        }
        enc.endEncoding()
        commitAndWait(cb)

        state.position += S
        activeStepCounters?.tokensProcessed += S

        guard projectLogits else { return [] }
        return readLogits()
    }

    /// One DeltaNet layer over the whole chunk (S1b token batching): one
    /// batched input norm, then one batched dispatch per in_proj tensor
    /// covering all tokens, one batched deinterleave, then the chunked
    /// recurrence (one cross-token conv scan, batched decay/beta prep and
    /// q/k norms, one T=n gated scan, one batched gated norm), then one
    /// batched output projection and one batched residual add. One token
    /// degenerates to exactly the decode core's encode stream, keeping the
    /// per-token scan alive as the chunked path's oracle.
    private func encChunkDelta(
        _ enc: MTLComputeCommandEncoder, delta: DeltaGPU, fl: FastLayer, tokens n: Int
    ) throws {
        try encChunkNormCopy(enc, w: fl.inputNorm, dstOff: reg.x0, tokens: n)
        barrier(enc)
        for proj in deltaInProjections(delta) {
            try engine.encodeGemvBatch(
                enc, proj.lin, x: prefillScratchBuf!, y: prefillScratchBuf!,
                slots: chunkGemvSlots(n, src: reg.x0, dst: proj.dst))
        }
        barrier(enc)
        if config.deltaLayout == .fusedInterleaved {
            try encChunkDeinterleave(enc, tokens: n)
            barrier(enc)
        }
        if n == 1 {
            try encDeltaRecurrence(enc, fl: fl, slot: prefillSlot(0))
        } else {
            try encChunkDeltaRecurrence(enc, fl: fl, tokens: n)
        }
        try engine.encodeGemvBatch(
            enc, delta.outProj, x: prefillScratchBuf!, y: prefillScratchBuf!,
            slots: chunkGemvSlots(n, src: reg.dn, dst: reg.r))
        barrier(enc)
        try encChunkResidualAdd(enc, tokens: n)
        barrier(enc)
    }

    /// The chunk's recurrence stage as pure cross-token dispatches (S1b
    /// chunked recurrence): one conv_scan advances the layer's shared conv
    /// history through all n tokens in-dispatch — its step loop is the
    /// chained per-token conv_steps' arithmetic verbatim, history carried
    /// in registers instead of a barrier-fenced device round trip per
    /// token; decay/beta prep and the weightless q/k norms are independent
    /// per token and encode once per chunk through their stride twins; one
    /// delta_gather packs the normed rows into the contiguous [T, row]
    /// staging blocks, a single T=n gated_delta_step replaces the n
    /// per-token scans, and one batched gated norm reads the scan's staged y
    /// rows. Every batched row is the single-token kernel's arithmetic
    /// verbatim and both scans' internal step loops perform the same
    /// ascending-order arithmetic the per-token dispatches did — state
    /// carried in f32 registers instead of a per-token f32 device round trip
    /// — so every conv row, y row, and both final states are bitwise
    /// identical. Nothing here dispatches per token any more. Ends on a
    /// barrier like the decode core.
    private func encChunkDeltaRecurrence(
        _ enc: MTLComputeCommandEncoder, fl: FastLayer, tokens n: Int
    ) throws {
        let cfg = config
        let valueDim = cfg.valueDim
        try engine.encodeConvScan(
            enc, data: prefillScratchBuf!, hist: fl.hist!, weight: fl.convW!,
            convDim: cfg.convDim, K: cfg.linearConvKernelDim,
            inOff: reg.qkv, outOff: reg.conv, slotStride: reg.total, tokens: n)
        barrier(enc)   // the in-place q/k norms read the scan's conv rows
        // Reads the slots' a/b projection rows (behind encChunkDelta's
        // barrier) and writes gb, disjoint from the conv rows the norms
        // touch in place — one barrier below covers both for the gather.
        try engine.encodeDeltaPreBatch(
            enc, data: prefillScratchBuf!, aLog: fl.aLog!, dtBias: fl.dtBias!,
            nv: cfg.linearNumValueHeads, aOff: reg.a, bOff: reg.b, outOff: reg.gb,
            slotStride: reg.total, tokens: n)
        let nk = cfg.linearNumKeyHeads, dk = cfg.linearKeyHeadDim
        let invScale = 1 / Float(dk).squareRoot()
        try engine.encodeRMSNormRowsBatch(
            enc, data: prefillScratchBuf!, weight: nil, off: reg.conv,
            rows: nk, dim: dk, scale: invScale * invScale, eps: 1e-6,
            slotStride: reg.total, tokens: n)
        try engine.encodeRMSNormRowsBatch(
            enc, data: prefillScratchBuf!, weight: nil, off: reg.conv + cfg.keyDim,
            rows: nk, dim: dk, scale: invScale, eps: 1e-6,
            slotStride: reg.total, tokens: n)
        barrier(enc)
        try engine.encodeDeltaGather(
            enc, data: prefillScratchBuf!,
            keyDim: config.keyDim, valueDim: valueDim, nv: config.linearNumValueHeads,
            slotStride: reg.total, convOff: reg.conv, gbOff: reg.gb,
            qOut: prefillStage.q, kOut: prefillStage.k, vOut: prefillStage.v,
            gOut: prefillStage.g, bOut: prefillStage.b, tokens: n)
        barrier(enc)
        try encGatedDeltaScan(
            enc, fl: fl, steps: n, data: prefillScratchBuf!,
            qOff: prefillStage.q, kOff: prefillStage.k, vOff: prefillStage.v,
            gOff: prefillStage.g, bOff: prefillStage.b, yOff: prefillStage.y)
        barrier(enc)
        try engine.encodeGatedNormMulBatch(
            enc, data: prefillScratchBuf!, weight: fl.deltaNormW!,
            nv: cfg.linearNumValueHeads, dv: cfg.linearValueHeadDim,
            eps: Float(cfg.rmsNormEps),
            yOff: prefillStage.y, yStride: valueDim,
            zOff: reg.z, zStride: reg.total,
            outOff: reg.dn, outStride: reg.total, tokens: n)
        barrier(enc)
    }

    /// One batched post-attention norm over every chunk token, then the
    /// router logits as one token-batched dispatch.
    private func encChunkRouterProbe(
        _ enc: MTLComputeCommandEncoder, moeGate: GPULinear, fl: FastLayer, tokens n: Int
    ) throws {
        try encChunkNormCopy(enc, w: fl.postNorm, dstOff: reg.xmoe, tokens: n)
        barrier(enc)
        try engine.encodeGemvBatch(
            enc, moeGate, x: prefillScratchBuf!, y: prefillScratchBuf!,
            slots: chunkGemvSlots(n, src: reg.xmoe, dst: reg.rout))
    }

    /// One layer's MoE for a whole chunk. Each routed projection encodes as
    /// one token-batched dispatch per union expert covering every token
    /// routed to it, and each shared projection as one dispatch covering the
    /// whole chunk; outputs land in disjoint per-token slots so a stage runs
    /// fully in parallel, and each token's weighted accumulation uses the
    /// legacy pick order, keeping numerics identical to the token-major path.
    private func encodeChunkMoE(_ enc: MTLComputeCommandEncoder, _ pend: PendingChunkMoE) throws {
        let cfg = config
        let D = cfg.hiddenSize, inter = cfg.moeIntermediateSize
        let shInter = cfg.sharedExpertIntermediateSize
        let topK = cfg.numExpertsPerTok
        let moe = layers[pend.layer].moe
        let fl = fastLayers[pend.layer]
        let projs = expertProjs

        // expert -> [(token, pick index)] in ascending token order.
        var assignments: [Int: [(t: Int, ki: Int)]] = [:]
        for (t, entry) in pend.perToken.enumerated() {
            for (ki, pick) in entry.picks.enumerated() {
                assignments[pick.0, default: []].append((t, ki))
            }
        }

        func linears(for expert: Int) -> (g: GPULinear, u: GPULinear, d: GPULinear, e: Int) {
            if let projs, let buf = pend.buffers[expert] {
                return (projs.gate.linear(on: buf), projs.up.linear(on: buf),
                        projs.down.linear(on: buf), 0)
            }
            let st = moe.stacks!
            return (st.gate.lin, st.up.lin, st.down.lin, expert)
        }

        // Stage 1: routed gate/up, one token-batched dispatch per (expert,
        // projection); then the shared gate/up and shared-gate logit, each
        // one dispatch covering every chunk token.
        let S = pend.perToken.count
        for expert in pend.union {
            let lin = linears(for: expert)
            let assigned = assignments[expert] ?? []
            try engine.encodeGemvBatch(enc, lin.g, rows: inter,
                wExtra: lin.e * (moe.stacks?.gate.wStride ?? 0),
                sExtra: lin.e * (moe.stacks?.gate.sStride ?? 0),
                bExtra: lin.e * (moe.stacks?.gate.sStride ?? 0),
                x: prefillScratchBuf!, y: prefillScratchBuf!,
                slots: assigned.map { a in
                    (xOff: a.t * reg.total + reg.xmoe,
                     yOff: a.t * reg.total + reg.exp + a.ki * inter)
                })
            try engine.encodeGemvBatch(enc, lin.u, rows: inter,
                wExtra: lin.e * (moe.stacks?.up.wStride ?? 0),
                sExtra: lin.e * (moe.stacks?.up.sStride ?? 0),
                bExtra: lin.e * (moe.stacks?.up.sStride ?? 0),
                x: prefillScratchBuf!, y: prefillScratchBuf!,
                slots: assigned.map { a in
                    let K = pend.perToken[a.t].picks.count
                    return (xOff: a.t * reg.total + reg.xmoe,
                            yOff: a.t * reg.total + reg.exp + K * inter + a.ki * inter)
                })
        }
        try engine.encodeGemvBatch(enc, moe.sharedGate, x: prefillScratchBuf!, y: prefillScratchBuf!,
            slots: chunkGemvSlots(S, src: reg.xmoe, dst: reg.sh))
        try engine.encodeGemvBatch(enc, moe.sharedUp, x: prefillScratchBuf!, y: prefillScratchBuf!,
            slots: chunkGemvSlots(S, src: reg.xmoe, dst: reg.sh + shInter))
        try engine.encodeGemvBatch(enc, fl.sharedGateLin, x: prefillScratchBuf!, y: prefillScratchBuf!,
            slots: chunkGemvSlots(S, src: reg.xmoe, dst: reg.shg))
        barrier(enc)

        // Stage 2: silu(gate) * up for every (token, pick) and shared chain.
        // Pick slot ki's rows sit at the same offsets in every token slot, so
        // each ki batches into one strided dispatch over the chunk (as does
        // the shared chain); a single-token chunk or a rare non-uniform pick
        // count keeps the legacy per-token loop.
        let uniformK = pend.perToken.allSatisfy { $0.picks.count == topK }
        if S > 1, uniformK {
            for ki in 0..<topK {
                try engine.encodeSiluMulBatch(enc, buf: prefillScratchBuf!, count: inter,
                    gOff: reg.exp + ki * inter,
                    uOff: reg.exp + topK * inter + ki * inter,
                    dstOff: reg.exp + 2 * topK * inter + ki * inter,
                    stride: reg.total, tokens: S)
            }
            try engine.encodeSiluMulBatch(enc, buf: prefillScratchBuf!, count: shInter,
                gOff: reg.sh, uOff: reg.sh + shInter, dstOff: reg.sh + 2 * shInter,
                stride: reg.total, tokens: S)
        } else {
            for (t, entry) in pend.perToken.enumerated() {
                let slot = prefillSlot(t)
                let K = entry.picks.count
                for ki in 0..<K {
                    try engine.encodeSiluMul(enc, buf: slot.scratch, count: inter,
                        gOff: slot.base + reg.exp + ki * inter,
                        uOff: slot.base + reg.exp + K * inter + ki * inter,
                        dstOff: slot.base + reg.exp + 2 * K * inter + ki * inter)
                }
                try engine.encodeSiluMul(enc, buf: slot.scratch, count: shInter,
                    gOff: slot.base + reg.sh, uOff: slot.base + reg.sh + shInter,
                    dstOff: slot.base + reg.sh + 2 * shInter)
            }
        }
        barrier(enc)

        // Stage 3: down projections, one token-batched dispatch per union
        // expert, plus the shared down covering every chunk token.
        for expert in pend.union {
            let lin = linears(for: expert)
            try engine.encodeGemvBatch(enc, lin.d, rows: D,
                wExtra: lin.e * (moe.stacks?.down.wStride ?? 0),
                sExtra: lin.e * (moe.stacks?.down.sStride ?? 0),
                bExtra: lin.e * (moe.stacks?.down.sStride ?? 0),
                x: prefillScratchBuf!, y: prefillScratchBuf!,
                slots: (assignments[expert] ?? []).map { a in
                    let K = pend.perToken[a.t].picks.count
                    return (xOff: a.t * reg.total + reg.exp + 2 * K * inter + a.ki * inter,
                            yOff: a.t * reg.total + reg.dexp + a.ki * D)
                })
        }
        try engine.encodeGemvBatch(enc, moe.sharedDown, x: prefillScratchBuf!, y: prefillScratchBuf!,
            slots: chunkGemvSlots(S, src: reg.sh + 2 * shInter, dst: reg.sh + 3 * shInter))
        barrier(enc)

        // Stage 4: weighted accumulation, one batched dispatch over the
        // chunk with a token-major weights table. Each token's K-expert
        // accumulation order stays inside the kernel, exactly the legacy
        // per-token loop, so batching cannot reorder any float sum.
        if S > 1, uniformK {
            var weights: [Float] = []
            weights.reserveCapacity(S * topK)
            for entry in pend.perToken { weights.append(contentsOf: entry.weights) }
            try engine.encodeWeightedAccumBatch(
                enc, h: prefillHiddenBuf!, hStride: D, data: prefillScratchBuf!,
                count: D, k: topK, dBase: reg.dexp, shOff: reg.sh + 3 * shInter,
                gateOff: reg.shg, stride: reg.total, weights: weights, tokens: S)
        } else {
            for (t, entry) in pend.perToken.enumerated() {
                try encWeightedAccum(enc, weights: entry.weights, count: entry.picks.count,
                                     slot: prefillSlot(t))
            }
        }
        barrier(enc)
    }
}
