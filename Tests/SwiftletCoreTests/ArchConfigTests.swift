import Testing
@testable import SwiftletCore

@Suite struct ArchConfigTests {
    @Test func qwen3Next80BLayout() {
        let c = ArchConfig.qwen3Next80B
        #expect(c.linearLayerCount == 36)
        #expect(c.fullAttentionLayerCount == 12)
        // mlx-lm pattern: layers 3, 7, 11, ... are full attention (0-based).
        #expect(!c.isLinearLayer(3) && !c.isLinearLayer(47))
        #expect(c.isLinearLayer(0) && c.isLinearLayer(1) && c.isLinearLayer(2) && c.isLinearLayer(4))
        #expect(c.expertParamCount == 3 * 2048 * 512)
        // ~1.77 MB per expert blob at int4 g64, ~44 GB pool.
        #expect(c.expertBlobBytesInt4G64 == 1_769_472)
        let poolGiB = Double(c.expertBlobBytesInt4G64 * c.routedExpertTotal) / Double(1 << 30)
        #expect(poolGiB > 40 && poolGiB < 45)
        // Per-token cold IO ~810 MiB.
        let coldMiB = Double(c.expertBlobBytesInt4G64 * c.routedFetchesPerToken) / Double(1 << 20)
        #expect(coldMiB > 700 && coldMiB < 900)
    }

    @Test func deltaNetStateIsContextIndependent() {
        let c = ArchConfig.qwen3Next80B
        // 36 layers x 32 heads x 128x128 f32 = 72 MiB exactly.
        #expect(c.deltaNetStateBytes == 36 * 32 * 128 * 128 * 4)
        // KV only exists on the 12 GQA layers: 12 x 2 x 256 x 2 x 2 = 24 KiB/token.
        #expect(c.kvBytesPerToken == 24_576)
    }

    @Test func familyVariants() {
        #expect(ArchConfig.qwen3_5_397B.linearLayerCount == 45)
        #expect(ArchConfig.qwen3_6_35B.routedExpertTotal == 40 * 256)
    }

    /// qwen3_5_moe family pendant to qwen3Next80BLayout: constants verified
    /// against mlx-community/Qwen3.6-35B-A3B-4bit's config.json.
    @Test func qwen3_6_35BLayout() {
        let c = ArchConfig.qwen3_6_35B
        #expect(c.family == .qwen3_5Moe)
        #expect(c.linearLayerCount == 30)
        #expect(c.fullAttentionLayerCount == 10)
        // layer_types list in the config: every 4th layer is full attention,
        // which the interval-4 rule reproduces exactly (0-based 3, 7, ..., 39).
        #expect(!c.isLinearLayer(3) && !c.isLinearLayer(39))
        #expect(c.isLinearLayer(0) && c.isLinearLayer(38))
        #expect(c.vocabSize == 248_320)
        // Same expert geometry as the 80B: 2048 x 512 x3, ~1.69 MiB int4 g64.
        #expect(c.expertBlobBytesInt4G64 == 1_769_472)
        // top-8 of 256 experts, norm_topk_prob renormalization required.
        #expect(c.expertTopK == 8 && c.expertCount == 256)
        #expect(c.normTopKProb)
    }

    @Test func qwen3_5_397BLayout() {
        let c = ArchConfig.qwen3_5_397B
        #expect(c.family == .qwen3_5Moe)
        #expect(c.linearLayerCount == 45 && c.fullAttentionLayerCount == 15)
        // 3 x 4096 x 1024 params -> 6.75 MiB per expert at int4 g64.
        #expect(c.expertParamCount == 3 * 4096 * 1024)
        #expect(c.expertBlobBytesInt4G64 == 7_077_888)
        // ~202.5 GiB routed-expert pool (container ~207 GiB with the dense
        // part), ~3.96 GiB fetched per token.
        let poolGiB = Double(c.expertBlobBytesInt4G64 * c.routedExpertTotal) / Double(1 << 30)
        #expect(poolGiB > 200 && poolGiB < 210)
        let perTokenGiB = Double(c.expertBlobBytesInt4G64 * c.routedFetchesPerToken) / Double(1 << 30)
        #expect(perTokenGiB > 3.8 && perTokenGiB < 4.1)
    }
}
