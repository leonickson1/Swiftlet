import Foundation
import Testing
@testable import SwiftletCore

/// S1b: layer-major, chunked prefill with expert-union routing. The bar:
/// final logits, greedy continuation, and KV/recurrent state must match the
/// legacy token-major path under the S1a tolerance discipline; the experts
/// touched per layer must be exactly the S1b-a oracle's planned unions; and
/// the new schedule's buffer/dispatch shape is pinned as exactly as the S3a
/// baselines pin the old one. Decode steps never take this path.
@Suite struct LayerMajorPrefillTests {
    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    /// Layer-major token-batched baseline: one decode-token-shaped buffer
    /// sequence per chunk (the S2 9-buffer shape), with the dispatch cost
    /// decomposed into per-chunk batched dispatches (each dense/shared/
    /// router GEMV plus each batch twin or cross-token scan — norm copies,
    /// deinterleave, silu, residual adds, weighted accumulation, attention
    /// q prep / KV append / causal attend, conv scan, decay/beta prep, q/k
    /// norms, gated scan, gated norm — encodes once per chunk regardless of
    /// chunk size) and per-union-expert batched GEMVs (gate/up/down encode
    /// once per (expert, projection) per chunk). No dispatch scales with
    /// chunk size any more; the perTokenDispatches term is kept at zero as
    /// the regression tripwire for that property. A single-token chunk
    /// delegates every batch encode to the legacy per-token path and lands
    /// exactly on the pre-batching decomposition. Exact assertions, S3a
    /// style.
    struct LayerMajorBaseline {
        let commandBuffersPerChunk: Int
        /// Dispatches that stay per token inside a multi-token chunk.
        let perTokenDispatches: Int
        /// Batched dispatches paid once per multi-token chunk.
        let perChunkDispatches: Int
        let perUnionExpertDispatches: Int
        let lmHeadDispatches: Int
        /// The pre-batching decomposition (what a single-token chunk pays).
        let legacyPerTokenDispatches: Int
        let legacyPerChunkDispatches: Int
        /// The flat per-token cost of the unbatched schedule; chunk=1 must
        /// land exactly on it, larger chunks strictly below it.
        let unbatchedDispatchesPerToken: Int

        func chunkSizes(tokens: Int, chunkTokens: Int) -> [Int] {
            stride(from: 0, to: tokens, by: chunkTokens).map { min(chunkTokens, tokens - $0) }
        }

        func chunkCount(tokens: Int, chunkTokens: Int) -> Int {
            (tokens + chunkTokens - 1) / chunkTokens
        }

        func commandBuffers(tokens: Int, chunkTokens: Int) -> Int {
            commandBuffersPerChunk * chunkCount(tokens: tokens, chunkTokens: chunkTokens)
        }

        func dispatches(tokens: Int, chunkTokens: Int, unionSizes: [Int]) -> Int {
            chunkSizes(tokens: tokens, chunkTokens: chunkTokens).reduce(lmHeadDispatches) {
                $0 + ($1 == 1
                    ? legacyPerTokenDispatches + legacyPerChunkDispatches
                    : perTokenDispatches * $1 + perChunkDispatches)
            } + perUnionExpertDispatches * unionSizes.reduce(0, +)
        }
    }

    /// Scanning the conv steps across the chunk erases the last per-token
    /// dispatches from the layer-major decomposition.
    /// Old (glue twins + chunked scan + batched attend + batched prep
    /// chain): per token only the depthwise conv step advancing the layer's
    /// shared history — q4/q35 6 = 6x1 — with per chunk q4 170 = 66 GEMVs
    /// + 6x10 + 2x6 + 8x4 (q35: 176 = 78 + 6x9 + 2x6 + 8x4).
    /// New: the conv history rides across the chunk inside one conv_scan
    /// per delta layer (the kernel's step loop is the chained per-token
    /// steps' arithmetic verbatim, so the chunk-size bitwise pins hold
    /// unchanged) — per token 0; no dispatch scales with chunk size. Per
    /// chunk the GEMVs plus every batched stage: q4 176 = 66 + 6x11 (conv
    /// scan, input norm, deinterleave, decay/beta prep, q norm, k norm,
    /// gather, T-step scan, gated norm, residual, post norm) + 2x6 (input
    /// norm, q prep, KV append, attend, residual, post norm) + 8x4 (top-2
    /// silus, shared silu, weighted accum); q35 182 = 78 + 6x10 (no
    /// deinterleave) + 2x6 + 8x4.
    /// Per union expert 3 (gate/up/down). A single-token chunk delegates to
    /// the legacy path and lands exactly on the old pin: 104 + 66 + 3x16
    /// = 218 and 98 + 78 + 3x16 = 224 (16 = 8 layers x top-2 picks).
    static let q4Baseline = LayerMajorBaseline(
        commandBuffersPerChunk: MetalModelTests.commandBuffersPerToken,
        perTokenDispatches: 0,
        perChunkDispatches: 176,
        perUnionExpertDispatches: 3,
        lmHeadDispatches: 2,
        legacyPerTokenDispatches: 104,
        legacyPerChunkDispatches: 66,
        unbatchedDispatchesPerToken: 218
    )
    static let q35Baseline = LayerMajorBaseline(
        commandBuffersPerChunk: MetalModelTests.commandBuffersPerToken,
        perTokenDispatches: 0,
        perChunkDispatches: 182,
        perUnionExpertDispatches: 3,
        lmHeadDispatches: 2,
        legacyPerTokenDispatches: 98,
        legacyPerChunkDispatches: 78,
        unbatchedDispatchesPerToken: 224
    )

    static func argmax(_ v: [Float]) -> Int {
        var b = 0
        for i in 1..<v.count where v[i] > v[b] { b = i }
        return b
    }

    /// The shipped default is the S1b schedule with a bounded chunk.
    @Test func defaultPrefillModeIsLayerMajor() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let model = try QwenMetalModel(modelDir: dir)
        #expect(model.prefillMode == .layerMajor(chunkTokens: 32))
    }

    static func compare(
        _ modelName: String,
        chunkTokens: Int,
        baseline: LayerMajorBaseline,
        decodeBaseline: MetalModelTests.FastPathBaseline
    ) throws {
        let dir = fixturesDir.appendingPathComponent(modelName)
        let label = "\(modelName) chunk=\(chunkTokens)"
        let tokens = [1, 5, 9, 42, 7]

        let cpu = try QwenCPUModel(modelDir: dir)
        cpu.retainAllLayers = true
        let cpuState = cpu.makeQwenContext()
        var cpuLogits: [Float] = []
        for t in tokens { cpuLogits = try cpu.step([t], context: cpuState) }

        // Reference: the decode loop, token at a time (never layer-major).
        let sequentialGPU = try QwenMetalModel(modelDir: dir)
        let sequentialState = sequentialGPU.makeQwenContext()
        var sequentialLogits: [Float] = []
        for t in tokens { sequentialLogits = try sequentialGPU.step([t], context: sequentialState) }

        let layerMajorGPU = try QwenMetalModel(modelDir: dir)
        layerMajorGPU.prefillMode = .layerMajor(chunkTokens: chunkTokens)
        let layerMajorState = layerMajorGPU.makeQwenContext()
        var unionSizes: [Int] = []
        layerMajorGPU.prefillExpertUnionObserver = { unionSizes.append($1.count) }
        let layerMajorLogits = try layerMajorGPU.step(tokens, context: layerMajorState)
        layerMajorGPU.prefillExpertUnionObserver = nil
        let metrics = layerMajorGPU.lastStepMetrics

        // Correctness bar: same logits, same greedy pick, same KV state.
        let cpuDiff = MetalModelTests.maxAbsDiff(cpuLogits, layerMajorLogits)
        #expect(cpuDiff < 2e-3, "\(label): CPU vs layer-major maxAbsDiff \(cpuDiff)")
        let seqDiff = MetalModelTests.maxAbsDiff(sequentialLogits, layerMajorLogits)
        #expect(seqDiff < 2e-3, "\(label): sequential vs layer-major maxAbsDiff \(seqDiff)")
        MetalModelTests.expectMatchingKV(
            sequentialState, layerMajorState, label: "\(label) prompt")
        #expect(argmax(cpuLogits) == argmax(layerMajorLogits), "\(label): greedy diverged")

        // S1a property preserved: one LM head, intermediates elided.
        #expect(metrics.tokensProcessed == tokens.count, "\(label): token count")
        #expect(metrics.logitProjections == 1, "\(label): LM-head count")
        #expect(metrics.avoidedLogitProjections == tokens.count - 1, "\(label): elision count")
        MetalModelTests.expectInstrumentation(metrics, tokens: tokens.count, label: label)

        // S1b token-batched baselines, pinned exactly: buffers per chunk,
        // the dispatch decomposition, and the chunk-shaped phase timeline.
        let chunks = baseline.chunkCount(tokens: tokens.count, chunkTokens: chunkTokens)
        #expect(metrics.commandBuffersCommitted
                == baseline.commandBuffers(tokens: tokens.count, chunkTokens: chunkTokens),
                "\(label): layer-major command-buffer baseline changed")
        #expect(metrics.computeDispatchesEncoded == baseline.dispatches(
                    tokens: tokens.count, chunkTokens: chunkTokens, unionSizes: unionSizes),
                "\(label): layer-major dispatch baseline changed")
        // The point of token batching: never above the unbatched schedule,
        // strictly below it whenever a chunk holds more than one token.
        let unbatched = baseline.unbatchedDispatchesPerToken * tokens.count
            + baseline.lmHeadDispatches
        if chunkTokens > 1 {
            #expect(metrics.computeDispatchesEncoded < unbatched,
                    "\(label): token batching did not reduce dispatches")
        } else {
            #expect(metrics.computeDispatchesEncoded == unbatched,
                    "\(label): single-token chunks moved off the decode-shaped cost")
        }
        MetalModelTests.expectPhaseTimeline(
            metrics,
            expectedPhases: MetalModelTests.expectedTimelinePhases(
                config: layerMajorGPU.config, tokens: chunks),
            label: label
        )

        // Continuation decode after the layer-major prompt: same next logits
        // as the sequential state, and the decode baselines untouched.
        let continuation = 11
        let seqCont = try sequentialGPU.step([continuation], context: sequentialState)
        let lmCont = try layerMajorGPU.step([continuation], context: layerMajorState)
        let contDiff = MetalModelTests.maxAbsDiff(seqCont, lmCont)
        #expect(contDiff < 2e-3, "\(label): continuation maxAbsDiff \(contDiff)")
        #expect(argmax(seqCont) == argmax(lmCont), "\(label): continuation greedy diverged")
        MetalModelTests.expectMatchingKV(
            sequentialState, layerMajorState, label: "\(label) continuation")
        MetalModelTests.expectFastPathBaseline(
            layerMajorGPU.lastStepMetrics, tokens: 1,
            baseline: decodeBaseline, label: "\(label) continuation"
        )
    }

    @Test func layerMajorMatchesSequentialOnQuantizedTiny() throws {
        try Self.compare("tiny-model-q4", chunkTokens: 32,
                         baseline: Self.q4Baseline, decodeBaseline: MetalModelTests.q4Baseline)
    }

    @Test func layerMajorMatchesSequentialOnQwen35Tiny() throws {
        try Self.compare("tiny-model-q35", chunkTokens: 32,
                         baseline: Self.q35Baseline, decodeBaseline: MetalModelTests.q35Baseline)
    }

    /// A prompt longer than the chunk crosses chunk boundaries with the same
    /// math: chunks of 2/2/1 must reproduce the sequential state exactly.
    @Test func chunkedPrefillSpansThePrompt() throws {
        try Self.compare("tiny-model-q4", chunkTokens: 2,
                         baseline: Self.q4Baseline, decodeBaseline: MetalModelTests.q4Baseline)
    }

    /// chunkTokens=1 degenerates to token order — the boundary case where
    /// layer-major and token-major visit (token, layer) identically.
    @Test func singleTokenChunksDegenerateToTokenOrder() throws {
        try Self.compare("tiny-model-q4", chunkTokens: 1,
                         baseline: Self.q4Baseline, decodeBaseline: MetalModelTests.q4Baseline)
    }

    /// The split DeltaNet layout (q35) must batch across chunk boundaries
    /// too: 4 in_proj dispatches per chunk instead of per token.
    @Test func chunkedPrefillSpansThePromptOnQwen35() throws {
        try Self.compare("tiny-model-q35", chunkTokens: 2,
                         baseline: Self.q35Baseline, decodeBaseline: MetalModelTests.q35Baseline)
    }

    /// Token batching is pure reordering: chunk sizes 1, 2, and 5 must
    /// produce bitwise-identical final logits, because each batched GEMV row
    /// computes exactly the unbatched kernel's arithmetic and the recurrent
    /// stages keep ascending token order.
    @Test(arguments: ["tiny-model-q4", "tiny-model-q35"])
    func chunkSizeIsBitwiseIrrelevant(modelName: String) throws {
        let dir = Self.fixturesDir.appendingPathComponent(modelName)
        let tokens = [1, 5, 9, 42, 7]
        var reference: [Float] = []
        for chunk in [1, 2, 5] {
            let model = try QwenMetalModel(modelDir: dir)
            model.prefillMode = .layerMajor(chunkTokens: chunk)
            let state = model.makeQwenContext()
            let logits = try model.step(tokens, context: state)
            if reference.isEmpty {
                reference = logits
            } else {
                #expect(logits.map(\.bitPattern) == reference.map(\.bitPattern),
                        "\(modelName): chunk=\(chunk) logits diverge bitwise from chunk=1")
            }
        }
    }

    // MARK: - Expert unions vs the S1b-a oracle

    /// The layer-major path must route every (token, layer) exactly as the
    /// token loop does, and fetch per layer exactly the union the
    /// PrefillExpertUnionPlan oracle computes from those routes — per chunk.
    static func expectUnionsMatchPlan(_ modelName: String, chunkTokens: Int) throws {
        let dir = fixturesDir.appendingPathComponent(modelName)
        let label = "\(modelName) chunk=\(chunkTokens)"
        let tokens = [1, 5, 9, 42, 7]

        // Oracle input: token-major routes recorded from the decode loop.
        let loopModel = try QwenMetalModel(modelDir: dir)
        let layerCount = loopModel.config.numHiddenLayers
        var loopFlat: [(layer: Int, experts: [Int])] = []
        loopModel.routedExpertObserver = { loopFlat.append(($0, $1)) }
        let loopState = loopModel.makeQwenContext()
        for t in tokens { _ = try loopModel.step([t], context: loopState) }
        loopModel.routedExpertObserver = nil
        #expect(loopFlat.count == tokens.count * layerCount, "\(label): observation shape")
        var loopRoutes: [[[Int]]] = []
        for t in 0..<tokens.count {
            var perLayer: [[Int]] = []
            for l in 0..<layerCount {
                let entry = loopFlat[t * layerCount + l]
                #expect(entry.layer == l, "\(label): loop layer order")
                perLayer.append(entry.experts)
            }
            loopRoutes.append(perLayer)
        }

        // The layer-major prefill records what it routed and what it fetched.
        let model = try QwenMetalModel(modelDir: dir)
        model.prefillMode = .layerMajor(chunkTokens: chunkTokens)
        var routed: [(layer: Int, experts: [Int])] = []
        var unions: [(layer: Int, experts: [Int])] = []
        model.routedExpertObserver = { routed.append(($0, $1)) }
        model.prefillExpertUnionObserver = { unions.append(($0, $1)) }
        let state = model.makeQwenContext()
        _ = try model.step(tokens, context: state)
        model.routedExpertObserver = nil
        model.prefillExpertUnionObserver = nil

        // Expected order: per chunk, per layer — every token's route (in
        // token order), then that layer's planned union.
        var expectedRouted: [(Int, [Int])] = []
        var expectedUnions: [(Int, [Int])] = []
        var start = 0
        while start < tokens.count {
            let end = min(start + chunkTokens, tokens.count)
            let chunkPlan = try PrefillExpertUnionPlan(
                tokenRoutes: Array(loopRoutes[start..<end]),
                expertCount: model.config.numExperts
            )
            for l in 0..<layerCount {
                for t in start..<end { expectedRouted.append((l, loopRoutes[t][l])) }
                expectedUnions.append((l, chunkPlan.layers[l].experts))
            }
            start = end
        }
        #expect(routed.map(\.layer) == expectedRouted.map(\.0),
                "\(label): layer-major routing order diverged")
        #expect(routed.map(\.experts) == expectedRouted.map(\.1),
                "\(label): layer-major routing diverged from the token loop")
        #expect(unions.map(\.layer) == expectedUnions.map(\.0),
                "\(label): union layer order diverged")
        #expect(unions.map(\.experts) == expectedUnions.map(\.1),
                "\(label): fetched unions diverge from the oracle plan")

        // Single chunk: the fetched unions are exactly the whole-prompt plan
        // replayed layer-major (the S1b-a equivalence pattern, now consumed
        // by the real schedule).
        if chunkTokens >= tokens.count {
            let plan = try PrefillExpertUnionPlan(
                tokenRoutes: loopRoutes, expertCount: model.config.numExperts)
            var replayed: [(Int, [Int])] = []
            plan.replayLayerMajor { replayed.append(($0, $1)) }
            #expect(unions.map(\.layer) == replayed.map(\.0)
                    && unions.map(\.experts) == replayed.map(\.1),
                    "\(label): schedule does not consume the plan verbatim")
        }
    }

    @Test func unionsMatchThePlanOnQuantizedTiny() throws {
        try Self.expectUnionsMatchPlan("tiny-model-q4", chunkTokens: 32)
    }

    @Test func unionsMatchThePlanOnQwen35Tiny() throws {
        try Self.expectUnionsMatchPlan("tiny-model-q35", chunkTokens: 32)
    }

    @Test func chunkedUnionsMatchPerChunkPlans() throws {
        try Self.expectUnionsMatchPlan("tiny-model-q4", chunkTokens: 2)
    }

    // MARK: - Streaming (qpack) path

    /// On the streaming container the layer-major prefill must stay exact and
    /// its expert-cache traffic must be exactly one union fetch per layer:
    /// the cache sees sum(union sizes) requests for the whole prompt, not
    /// tokens x top-k.
    @Test func layerMajorQpackStreamingMatchesSequential() throws {
        let src = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-q4-s1b-\(UUID().uuidString).qpack")
        defer { try? FileManager.default.removeItem(at: out) }
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        try repacker.repack()

        let tokens = [1, 5, 9, 42, 7, 99]
        let sequentialGPU = try QwenMetalModel(modelDir: out, cacheBudgetGB: 0.05)
        let layerMajorGPU = try QwenMetalModel(modelDir: out, cacheBudgetGB: 0.05)
        #expect(sequentialGPU.expertCache != nil, "qpack mode not engaged")
        #expect(layerMajorGPU.expertCache != nil, "qpack mode not engaged")

        let sequentialState = sequentialGPU.makeQwenContext()
        var sequentialLogits: [Float] = []
        for t in tokens { sequentialLogits = try sequentialGPU.step([t], context: sequentialState) }

        var unionSizes: [Int] = []
        layerMajorGPU.prefillExpertUnionObserver = { unionSizes.append($1.count) }
        let cache = layerMajorGPU.expertCache!
        let requestsBefore = cache.hits + cache.misses
        let layerMajorState = layerMajorGPU.makeQwenContext()
        let layerMajorLogits = try layerMajorGPU.step(tokens, context: layerMajorState)
        layerMajorGPU.prefillExpertUnionObserver = nil
        let requests = cache.hits + cache.misses - requestsBefore

        let diff = MetalModelTests.maxAbsDiff(sequentialLogits, layerMajorLogits)
        #expect(diff < 2e-3, "qpack layer-major maxAbsDiff \(diff)")
        MetalModelTests.expectMatchingKV(
            sequentialState, layerMajorState, label: "qpack layer-major prompt")
        #expect(layerMajorGPU.lastStepMetrics.logitProjections == 1)
        #expect(layerMajorGPU.lastStepMetrics.commandBuffersCommitted
                == Self.q4Baseline.commandBuffers(tokens: tokens.count, chunkTokens: 32),
                "qpack layer-major command-buffer baseline changed")
        #expect(layerMajorGPU.lastStepMetrics.computeDispatchesEncoded
                == Self.q4Baseline.dispatches(
                    tokens: tokens.count, chunkTokens: 32, unionSizes: unionSizes),
                "qpack layer-major dispatch baseline changed")

        #expect(unionSizes.count == layerMajorGPU.config.numHiddenLayers,
                "one union fetch per layer expected")
        #expect(requests == unionSizes.reduce(0, +),
                "expert-cache traffic is not one union fetch per layer")
        let tokenMajorRequests = tokens.count
            * layerMajorGPU.config.numHiddenLayers
            * layerMajorGPU.config.numExpertsPerTok
        #expect(requests <= tokenMajorRequests,
                "union fetches exceed token-major traffic")

        let continuation = 11
        let seqCont = try sequentialGPU.step([continuation], context: sequentialState)
        let lmCont = try layerMajorGPU.step([continuation], context: layerMajorState)
        let contDiff = MetalModelTests.maxAbsDiff(seqCont, lmCont)
        #expect(contDiff < 2e-3, "qpack layer-major continuation maxAbsDiff \(contDiff)")
        MetalModelTests.expectMatchingKV(
            sequentialState, layerMajorState, label: "qpack layer-major continuation")
    }
}
