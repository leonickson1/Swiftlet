import Foundation
import Testing
@testable import SwiftletCore

/// The Metal runtime must reproduce the CPU reference: same greedy tokens,
/// near-identical logits, on both quantized (int4) and plain (f32) tiny models
/// and both DeltaNet layouts.
@Suite struct MetalModelTests {
    /// S3a whole-step aggregate regression baseline. These counts intentionally
    /// do not assign work to phases or make timeline/performance claims.
    struct FastPathBaseline {
        let commandBuffersPerToken: Int
        let intermediateDispatches: Int
        let finalDispatches: Int

        func commandBuffers(tokens: Int) -> Int {
            commandBuffersPerToken * tokens
        }

        func dispatches(tokens: Int) -> Int {
            guard tokens > 0 else { return 0 }
            return intermediateDispatches * (tokens - 1) + finalDispatches
        }
    }

    /// S2: moving attention decode on-GPU merged each attention layer's two
    /// command buffers (projections / CPU round trip / finish+router) into
    /// one, so a token costs one buffer per layer plus the tail. Was 11 with
    /// the CPU attention core (8 layers, 2 of them attention).
    static let commandBuffersPerToken = 9
    /// S2 dispatch deltas: +3 per attention layer per token (q prep, KV
    /// append, causal attention). Was 212/214 (q4) and 218/220 (q35) when
    /// the attention core ran on CPU.
    static let q4Baseline = FastPathBaseline(
        commandBuffersPerToken: commandBuffersPerToken,
        intermediateDispatches: 218,
        finalDispatches: 220
    )
    static let q35Baseline = FastPathBaseline(
        commandBuffersPerToken: commandBuffersPerToken,
        intermediateDispatches: 224,
        finalDispatches: 226
    )

    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    static func maxAbsDiff(_ lhs: [Float], _ rhs: [Float]) -> Float {
        guard lhs.count == rhs.count else { return .infinity }
        var result: Float = 0
        for i in lhs.indices { result = max(result, abs(lhs[i] - rhs[i])) }
        return result
    }

    static func expectMatchingKV(
        _ lhs: QwenInferenceContext, _ rhs: QwenInferenceContext,
        label: String
    ) {
        #expect(lhs.position == rhs.position, "\(label): positions diverged")
        #expect(Set(lhs.kv.keys) == Set(rhs.kv.keys), "\(label): KV layers diverged")
        for layer in lhs.kv.keys {
            guard let l = lhs.kv[layer], let r = rhs.kv[layer] else { continue }
            #expect(l.k.count == r.k.count, "\(label): layer \(layer) K size diverged")
            #expect(l.v.count == r.v.count, "\(label): layer \(layer) V size diverged")
            #expect(Self.maxAbsDiff(l.k, r.k) < 1e-6, "\(label): layer \(layer) K diverged")
            #expect(Self.maxAbsDiff(l.v, r.v) < 1e-6, "\(label): layer \(layer) V diverged")
        }
    }

    static func expectInstrumentation(
        _ metrics: QwenMetalModel.StepMetrics, tokens: Int, label: String
    ) {
        #expect(metrics.completedWithoutThrow, "\(label): step threw")
        #expect(metrics.tokensProcessed == tokens, "\(label): token count")
        #expect(metrics.logitProjections == 1, "\(label): LM-head count")
        #expect(metrics.commandBuffersCommitted > 0, "\(label): no command buffers")
        #expect(metrics.blockingWaits == metrics.commandBuffersCommitted,
                "\(label): commit/wait mismatch")
        #expect(metrics.commandBufferErrors == 0, "\(label): command-buffer error")
        #expect(metrics.computeDispatchesEncoded > 0, "\(label): no compute dispatches")
        #expect(metrics.gpuTimedCommandBuffers + metrics.gpuUntimedCommandBuffers
                + metrics.commandBufferErrors == metrics.commandBuffersCommitted,
                "\(label): GPU timing samples")
    }

    static func expectFastPathBaseline(
        _ metrics: QwenMetalModel.StepMetrics,
        tokens: Int,
        baseline: FastPathBaseline,
        label: String
    ) {
        #expect(metrics.commandBuffersCommitted == baseline.commandBuffers(tokens: tokens),
                "\(label): fast-path command-buffer baseline changed")
        #expect(metrics.computeDispatchesEncoded == baseline.dispatches(tokens: tokens),
                "\(label): fast-path dispatch baseline changed")
    }

    /// S3b: the committed command-buffer timeline must label every buffer,
    /// follow the expected schedule exactly, and account for the S3a
    /// aggregates without inventing timing the buffers never reported.
    /// Only meaningful for a step that completed without throwing.
    static func expectPhaseTimeline(
        _ metrics: QwenMetalModel.StepMetrics,
        expectedPhases: [[QwenMetalModel.StepPhase]],
        label: String
    ) {
        let timeline = metrics.commandBufferTimeline
        #expect(timeline.count == metrics.commandBuffersCommitted,
                "\(label): timeline misses committed buffers")
        #expect(timeline.map(\.phases) == expectedPhases,
                "\(label): phase labels diverged from the expected schedule")

        let phaseDispatchTotal = metrics.phaseDispatchesEncoded.values.reduce(0, +)
        #expect(phaseDispatchTotal == metrics.computeDispatchesEncoded,
                "\(label): phase dispatch totals leak dispatches")
        #expect(metrics.phaseDispatchesEncoded[.other, default: 0] == 0,
                "\(label): dispatches encoded outside every labeled phase")

        var waitSum = 0.0
        var gpuSum = 0.0
        var timed = 0
        var untimed = 0
        var failed = 0
        var timelineDispatches = 0
        for sample in timeline {
            #expect(!sample.phases.isEmpty, "\(label): unlabeled command buffer")
            #expect(sample.completed, "\(label): failed buffer in timeline")
            #expect(sample.encodeSeconds >= 0 && sample.waitSeconds >= 0,
                    "\(label): negative buffer timing")
            let phaseEncode = sample.phaseEncodeSeconds.values.reduce(0, +)
            #expect(phaseEncode <= sample.encodeSeconds + 1e-6,
                    "\(label): phase encode time exceeds the buffer's encode span")
            waitSum += sample.waitSeconds
            if !sample.completed {
                failed += 1
            } else if let gpu = sample.gpuSeconds {
                #expect(gpu >= 0, "\(label): negative GPU duration")
                gpuSum += gpu
                timed += 1
            } else {
                untimed += 1
            }
            timelineDispatches += sample.dispatchesEncoded
        }
        #expect(abs(waitSum - metrics.blockingWaitSeconds) < 1e-6,
                "\(label): timeline wait diverged from the S3a aggregate")
        #expect(abs(gpuSum - metrics.gpuExecutionSeconds) < 1e-6,
                "\(label): timeline GPU time diverged from the S3a aggregate")
        #expect(timed == metrics.gpuTimedCommandBuffers, "\(label): timed buffer count")
        #expect(untimed == metrics.gpuUntimedCommandBuffers, "\(label): untimed buffer count")
        #expect(failed == metrics.commandBufferErrors, "\(label): failed buffer count")
        #expect(timelineDispatches == metrics.computeDispatchesEncoded,
                "\(label): timeline dispatches diverged from the S3a aggregate")
        #expect(timeline.filter { $0.phases.contains(.lmHead) }.count == metrics.logitProjections,
                "\(label): LM-head buffer count")
    }

    /// The label sequence the current fast-path schedule must produce: per
    /// token, one buffer per layer (S2 merged the attention layers' former
    /// two-buffer CPU round trip), and a tail buffer that flushes the last
    /// layer's experts (adding the LM head only on the final token). Labels
    /// are in canonical declaration order.
    static func expectedTimelinePhases(
        config: QwenConfig, tokens: Int
    ) -> [[QwenMetalModel.StepPhase]] {
        var expected: [[QwenMetalModel.StepPhase]] = []
        for token in 0..<tokens {
            for layer in 0..<config.numHiddenLayers {
                let flushesMoE = layer > 0
                if config.isLinearLayer(layer) {
                    expected.append(flushesMoE ? [.delta, .moe, .router] : [.delta, .router])
                } else {
                    expected.append(flushesMoE ? [.attention, .moe, .router] : [.attention, .router])
                }
            }
            expected.append(token == tokens - 1 ? [.moe, .lmHead] : [.moe])
        }
        return expected
    }

    static func compare(_ modelName: String, baseline: FastPathBaseline) throws {
        let dir = fixturesDir.appendingPathComponent(modelName)
        let cpu = try QwenCPUModel(modelDir: dir)
        cpu.retainAllLayers = true
        let sequentialGPU = try QwenMetalModel(modelDir: dir)
        let elidingGPU = try QwenMetalModel(modelDir: dir)
        // These baselines pin the legacy token-major prompt schedule (S1a).
        // The layer-major S1b schedule has its own pinned baselines in
        // LayerMajorPrefillTests.
        sequentialGPU.prefillMode = .tokenMajor
        elidingGPU.prefillMode = .tokenMajor
        let tokens = [1, 5, 9, 42, 7]

        let cpuState = cpu.makeQwenContext()
        let sequentialState = sequentialGPU.makeQwenContext()
        var cpuLogits: [Float] = []
        var sequentialLogits: [Float] = []
        for t in tokens {
            cpuLogits = try cpu.step([t], context: cpuState)
            sequentialLogits = try sequentialGPU.step([t], context: sequentialState)
            #expect(sequentialGPU.lastStepMetrics.tokensProcessed == 1)
            #expect(sequentialGPU.lastStepMetrics.logitProjections == 1)
            #expect(sequentialGPU.lastStepMetrics.avoidedLogitProjections == 0)
        }
        let singleMetrics = sequentialGPU.lastStepMetrics
        Self.expectInstrumentation(singleMetrics, tokens: 1, label: "\(modelName) one token")
        Self.expectFastPathBaseline(
            singleMetrics, tokens: 1, baseline: baseline, label: "\(modelName) one token"
        )
        Self.expectPhaseTimeline(
            singleMetrics,
            expectedPhases: Self.expectedTimelinePhases(config: sequentialGPU.config, tokens: 1),
            label: "\(modelName) one token"
        )

        let maxDiff = Self.maxAbsDiff(cpuLogits, sequentialLogits)
        #expect(maxDiff < 2e-3, "\(modelName): GPU vs CPU logits maxAbsDiff \(maxDiff)")

        // S1a intermediate LM-head elision must retain token-at-a-time output
        // and state. Separate instances isolate GPU-resident recurrence.
        let elidingState = elidingGPU.makeQwenContext()
        let elidingLogits = try elidingGPU.step(tokens, context: elidingState)
        let multiMetrics = elidingGPU.lastStepMetrics
        let elisionDiff = Self.maxAbsDiff(sequentialLogits, elidingLogits)
        #expect(elisionDiff < 2e-3, "\(modelName): LM-head elision logits maxAbsDiff \(elisionDiff)")
        #expect(elidingGPU.lastStepMetrics.tokensProcessed == tokens.count)
        #expect(elidingGPU.lastStepMetrics.logitProjections == 1)
        #expect(elidingGPU.lastStepMetrics.avoidedLogitProjections == tokens.count - 1)
        Self.expectInstrumentation(multiMetrics, tokens: tokens.count, label: "\(modelName) multi token")
        Self.expectFastPathBaseline(
            multiMetrics, tokens: tokens.count, baseline: baseline, label: "\(modelName) multi token"
        )
        Self.expectPhaseTimeline(
            multiMetrics,
            expectedPhases: Self.expectedTimelinePhases(config: elidingGPU.config, tokens: tokens.count),
            label: "\(modelName) multi token"
        )
        Self.expectMatchingKV(sequentialState, elidingState, label: "\(modelName) elision input")

        let continuation = 11
        let sequentialContinuation = try sequentialGPU.step([continuation], context: sequentialState)
        let elidingContinuation = try elidingGPU.step([continuation], context: elidingState)
        let continuationDiff = Self.maxAbsDiff(sequentialContinuation, elidingContinuation)
        #expect(continuationDiff < 2e-3, "\(modelName): continuation maxAbsDiff \(continuationDiff)")
        #expect(elidingGPU.lastStepMetrics.tokensProcessed == 1)
        #expect(elidingGPU.lastStepMetrics.logitProjections == 1)
        #expect(elidingGPU.lastStepMetrics.avoidedLogitProjections == 0)
        Self.expectInstrumentation(
            elidingGPU.lastStepMetrics, tokens: 1, label: "\(modelName) continuation"
        )
        Self.expectFastPathBaseline(
            elidingGPU.lastStepMetrics, tokens: 1,
            baseline: baseline, label: "\(modelName) continuation"
        )
        Self.expectPhaseTimeline(
            elidingGPU.lastStepMetrics,
            expectedPhases: Self.expectedTimelinePhases(config: elidingGPU.config, tokens: 1),
            label: "\(modelName) continuation"
        )
        Self.expectMatchingKV(sequentialState, elidingState, label: "\(modelName) continuation")

        func argmax(_ v: [Float]) -> Int {
            var b = 0
            for i in 1..<v.count where v[i] > v[b] { b = i }
            return b
        }
        #expect(argmax(cpuLogits) == argmax(sequentialLogits), "\(modelName): greedy diverged")
    }

    @Test func gpuMatchesCPUOnQuantizedTiny() throws {
        try Self.compare("tiny-model-q4", baseline: Self.q4Baseline)
    }

    /// S3b: the phase/timeline instrumentation must label the whole step,
    /// stay within the step wall clock, and rebuild per step call rather than
    /// accumulate across calls.
    @Test func phaseTimelineAccountsForWholeStep() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let model = try QwenMetalModel(modelDir: dir)
        // Pins the legacy token-major prompt schedule (S1a).
        model.prefillMode = .tokenMajor
        let state = model.makeQwenContext()
        _ = try model.step([1, 5, 9], context: state)
        let multi = model.lastStepMetrics
        Self.expectInstrumentation(multi, tokens: 3, label: "timeline multi")
        Self.expectFastPathBaseline(
            multi, tokens: 3, baseline: Self.q4Baseline, label: "timeline multi"
        )
        Self.expectPhaseTimeline(
            multi,
            expectedPhases: Self.expectedTimelinePhases(config: model.config, tokens: 3),
            label: "timeline multi"
        )
        let encodeSum = multi.commandBufferTimeline.reduce(0.0) { $0 + $1.encodeSeconds }
        #expect(encodeSum + multi.blockingWaitSeconds <= multi.stepWallSeconds + 1e-3,
                "timeline multi: encode+wait exceeds the step wall clock")

        _ = try model.step([11], context: state)
        let single = model.lastStepMetrics
        Self.expectPhaseTimeline(
            single,
            expectedPhases: Self.expectedTimelinePhases(config: model.config, tokens: 1),
            label: "timeline continuation"
        )
        #expect(single.commandBufferTimeline.count == Self.commandBuffersPerToken,
                "timeline continuation: timeline accumulated across steps")
    }

    /// Full streaming path: repack tiny model to .qpack, run the GPU model in
    /// cache/pread mode, compare against the CPU reference on the raw dir.
    @Test func gpuQpackStreamingMatchesCPU() throws {
        let src = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-q4-gpu-\(UUID().uuidString).qpack")
        defer { try? FileManager.default.removeItem(at: out) }
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        try repacker.repack()

        let cpu = try QwenCPUModel(modelDir: src)
        cpu.retainAllLayers = true
        let sequentialGPU = try QwenMetalModel(modelDir: out, cacheBudgetGB: 0.05)
        let elidingGPU = try QwenMetalModel(modelDir: out, cacheBudgetGB: 0.05)
        // Pins the legacy token-major prompt schedule (S1a); the layer-major
        // qpack path is covered by LayerMajorPrefillTests.
        sequentialGPU.prefillMode = .tokenMajor
        elidingGPU.prefillMode = .tokenMajor
        #expect(sequentialGPU.expertCache != nil, "qpack mode not engaged")
        #expect(elidingGPU.expertCache != nil, "qpack mode not engaged")

        let tokens = [1, 5, 9, 42, 7, 99]
        let cpuState = cpu.makeQwenContext()
        let sequentialState = sequentialGPU.makeQwenContext()
        var cpuLogits: [Float] = []
        var sequentialLogits: [Float] = []
        for t in tokens {
            cpuLogits = try cpu.step([t], context: cpuState)
            sequentialLogits = try sequentialGPU.step([t], context: sequentialState)
            #expect(sequentialGPU.lastStepMetrics.tokensProcessed == 1)
            #expect(sequentialGPU.lastStepMetrics.logitProjections == 1)
            #expect(sequentialGPU.lastStepMetrics.avoidedLogitProjections == 0)
        }
        let singleMetrics = sequentialGPU.lastStepMetrics
        Self.expectInstrumentation(singleMetrics, tokens: 1, label: "qpack one token")
        Self.expectFastPathBaseline(
            singleMetrics, tokens: 1, baseline: Self.q4Baseline, label: "qpack one token"
        )
        Self.expectPhaseTimeline(
            singleMetrics,
            expectedPhases: Self.expectedTimelinePhases(config: sequentialGPU.config, tokens: 1),
            label: "qpack one token"
        )
        let maxDiff = Self.maxAbsDiff(cpuLogits, sequentialLogits)
        #expect(maxDiff < 2e-3, "qpack GPU vs CPU logits maxAbsDiff \(maxDiff)")

        let elidingState = elidingGPU.makeQwenContext()
        let elidingLogits = try elidingGPU.step(tokens, context: elidingState)
        let multiMetrics = elidingGPU.lastStepMetrics
        let elisionDiff = Self.maxAbsDiff(sequentialLogits, elidingLogits)
        #expect(elisionDiff < 2e-3, "qpack LM-head elision maxAbsDiff \(elisionDiff)")
        #expect(elidingGPU.lastStepMetrics.tokensProcessed == tokens.count)
        #expect(elidingGPU.lastStepMetrics.logitProjections == 1)
        #expect(elidingGPU.lastStepMetrics.avoidedLogitProjections == tokens.count - 1)
        Self.expectInstrumentation(multiMetrics, tokens: tokens.count, label: "qpack multi token")
        Self.expectFastPathBaseline(
            multiMetrics, tokens: tokens.count,
            baseline: Self.q4Baseline, label: "qpack multi token"
        )
        Self.expectPhaseTimeline(
            multiMetrics,
            expectedPhases: Self.expectedTimelinePhases(config: elidingGPU.config, tokens: tokens.count),
            label: "qpack multi token"
        )
        Self.expectMatchingKV(sequentialState, elidingState, label: "qpack elision input")

        let continuation = 11
        let sequentialContinuation = try sequentialGPU.step([continuation], context: sequentialState)
        let elidingContinuation = try elidingGPU.step([continuation], context: elidingState)
        let continuationDiff = Self.maxAbsDiff(sequentialContinuation, elidingContinuation)
        #expect(continuationDiff < 2e-3, "qpack continuation maxAbsDiff \(continuationDiff)")
        #expect(elidingGPU.lastStepMetrics.logitProjections == 1)
        Self.expectInstrumentation(
            elidingGPU.lastStepMetrics, tokens: 1, label: "qpack continuation"
        )
        Self.expectFastPathBaseline(
            elidingGPU.lastStepMetrics, tokens: 1,
            baseline: Self.q4Baseline, label: "qpack continuation"
        )
        Self.expectPhaseTimeline(
            elidingGPU.lastStepMetrics,
            expectedPhases: Self.expectedTimelinePhases(config: elidingGPU.config, tokens: 1),
            label: "qpack continuation"
        )
        Self.expectMatchingKV(sequentialState, elidingState, label: "qpack continuation")

        for cache in [sequentialGPU.expertCache!, elidingGPU.expertCache!] {
            #expect(cache.hits + cache.misses > 0)
        }
    }

    @Test func gpuMatchesCPUOnQwen35Tiny() throws {
        try Self.compare("tiny-model-q35", baseline: Self.q35Baseline)
    }
}
