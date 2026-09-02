import Foundation
import Testing
@testable import SwiftletCore

/// S2: full-attention decode runs on Metal against a GPU-resident KV cache.
/// state.kv stays the observable KV state — a CPU-side mirror copied from the
/// GPU rows after each attention layer — so the existing KV parity
/// assertions keep their meaning instead of trivially comparing empties.
@Suite struct MetalAttentionTests {
    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    /// After a decode loop, every attention layer must hold its KV rows on
    /// the GPU, the state.kv mirror must equal those rows bitwise (it is
    /// copied from them), and the mirror must match the CPU reference
    /// model's KV within accumulation noise. That noise is dominated by the
    /// k projection itself — quantized affine GEMV on GPU vs dequantized
    /// BLAS on CPU — not by the attention kernels; measured 1.2e-5 on
    /// M4 Max, pinned at 5e-5.
    @Test func decodeKVCacheIsGPUResidentAndMatchesCPU() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let cpu = try QwenCPUModel(modelDir: dir)
        cpu.retainAllLayers = true
        let gpu = try QwenMetalModel(modelDir: dir)
        let tokens = [1, 5, 9, 42, 7]
        let cpuState = cpu.makeQwenContext()
        let gpuState = gpu.makeQwenContext()
        for t in tokens {
            _ = try cpu.step([t], context: cpuState)
            _ = try gpu.step([t], context: gpuState)
        }

        var attentionLayers = 0
        for li in 0..<gpu.config.numHiddenLayers where !gpu.config.isLinearLayer(li) {
            attentionLayers += 1
            let rows = try #require(
                gpu.attentionKVCacheRows(layer: li, count: tokens.count),
                "layer \(li): no GPU-resident KV cache after decode"
            )
            let mirror = try #require(gpuState.kv[li], "layer \(li): KV mirror missing")
            #expect(rows.k == mirror.k, "layer \(li): GPU K rows diverge from the mirror")
            #expect(rows.v == mirror.v, "layer \(li): GPU V rows diverge from the mirror")

            let cpuKV = try #require(cpuState.kv[li], "layer \(li): CPU reference KV missing")
            let kDiff = MetalModelTests.maxAbsDiff(mirror.k, cpuKV.k)
            let vDiff = MetalModelTests.maxAbsDiff(mirror.v, cpuKV.v)
            #expect(kDiff < 5e-5, "layer \(li): K vs CPU reference maxAbsDiff \(kDiff)")
            #expect(vDiff < 5e-5, "layer \(li): V vs CPU reference maxAbsDiff \(vDiff)")
        }
        #expect(attentionLayers > 0, "fixture has no attention layers")

        // DeltaNet layers never grow a KV cache.
        for li in 0..<gpu.config.numHiddenLayers where gpu.config.isLinearLayer(li) {
            #expect(gpu.attentionKVCacheRows(layer: li, count: 1) == nil,
                    "layer \(li): DeltaNet layer grew a KV cache")
        }
    }

    /// Batched attention prefill must land the same GPU KV rows as the
    /// sequential decode loop — same kernels at the same absolute positions,
    /// so the rows match bitwise even across chunk boundaries — and the
    /// state.kv mirror must track the GPU rows through those boundaries.
    @Test func batchedPrefillKVMatchesSequentialDecode() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let tokens = [1, 5, 9, 42, 7]
        let sequential = try QwenMetalModel(modelDir: dir)
        let seqState = sequential.makeQwenContext()
        for t in tokens { _ = try sequential.step([t], context: seqState) }

        let chunked = try QwenMetalModel(modelDir: dir)
        chunked.prefillMode = .layerMajor(chunkTokens: 2) // crosses chunk boundaries
        let chunkState = chunked.makeQwenContext()
        _ = try chunked.step(tokens, context: chunkState)

        for li in 0..<chunked.config.numHiddenLayers where !chunked.config.isLinearLayer(li) {
            let seqRows = try #require(
                sequential.attentionKVCacheRows(layer: li, count: tokens.count),
                "layer \(li): sequential decode kept no GPU KV")
            let chunkRows = try #require(
                chunked.attentionKVCacheRows(layer: li, count: tokens.count),
                "layer \(li): batched prefill kept no GPU KV")
            #expect(seqRows.k == chunkRows.k, "layer \(li): batched K rows diverge from decode")
            #expect(seqRows.v == chunkRows.v, "layer \(li): batched V rows diverge from decode")
            let mirror = try #require(chunkState.kv[li], "layer \(li): chunk KV mirror missing")
            #expect(chunkRows.k == mirror.k, "layer \(li): chunk mirror diverges from GPU K rows")
            #expect(chunkRows.v == mirror.v, "layer \(li): chunk mirror diverges from GPU V rows")
        }
    }

    /// The decode schedule encodes each attention layer in one command
    /// buffer — projections, KV append, attention, out-proj, and the router
    /// probe together. The CPU round trip's second buffer is gone.
    @Test func attentionDecodeEncodesOneBufferPerLayer() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let model = try QwenMetalModel(modelDir: dir)
        let state = model.makeQwenContext()
        _ = try model.step([1], context: state)
        let timeline = model.lastStepMetrics.commandBufferTimeline
        #expect(timeline.count == model.config.numHiddenLayers + 1,
                "decode is not one buffer per layer plus the tail")

        let attentionLayerCount = (0..<model.config.numHiddenLayers)
            .filter { !model.config.isLinearLayer($0) }.count
        let attentionBuffers = timeline.filter { $0.phases.contains(.attention) }
        #expect(attentionBuffers.count == attentionLayerCount,
                "attention layers must encode exactly one buffer each")
        for sample in attentionBuffers {
            #expect(sample.phases.contains(.router),
                    "attention buffer no longer carries its router probe")
        }
    }
}
