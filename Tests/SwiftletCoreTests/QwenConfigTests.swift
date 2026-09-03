import Foundation
import Testing
@testable import SwiftletCore

/// Config-parsing regressions: absent keys must fall back to the same
/// FAMILY-SPECIFIC defaults mlx-lm's dataclasses use, not to one global value.
@Suite struct QwenConfigTests {
    static func write(_ json: [String: Any]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen-config-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: json)
        try data.write(to: dir.appendingPathComponent("config.json"))
        return dir.appendingPathComponent("config.json")
    }

    static func textConfigCommon() -> [String: Any] { [
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "vocab_size": 128,
        "max_position_embeddings": 512,
        "full_attention_interval": 4,
        "rope_parameters": [
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
        ],
    ] }

    /// Real Qwen3.5/3.6 checkpoints OMIT norm_topk_prob; mlx-lm's qwen3_5
    /// TextModelArgs defaults it to TRUE. Without renormalization every MoE
    /// output is scaled by the raw top-k softmax mass — the bug behind the
    /// corrupted qwen3_5_moe decodes.
    @Test func qwen35FamilyDefaultsNormTopkProbTrue() throws {
        let url = try Self.write([
            "model_type": "qwen3_5_moe",
            "text_config": Self.textConfigCommon(),
        ])
        let config = try QwenConfig(url: url)
        #expect(config.normTopkProb, "qwen3_5 family must renormalize top-k router weights by default")
        #expect(config.deltaLayout == .split)
        #expect(config.ropeTheta == 10_000_000)
        #expect(config.partialRotaryFactor == 0.25)
        #expect(config.maxPositionEmbeddings == 512)
    }

    /// qwen3_next's ModelArgs defaults norm_topk_prob to FALSE (the 80B
    /// checkpoint carries an explicit true). An absent key must stay false.
    @Test func qwen3NextDefaultsNormTopkProbFalse() throws {
        var flat: [String: Any] = Self.textConfigCommon()
        flat["model_type"] = "qwen3_next"
        flat.removeValue(forKey: "rope_parameters")
        flat["rope_theta"] = 10_000_000
        flat["partial_rotary_factor"] = 0.25
        let url = try Self.write(flat)
        let config = try QwenConfig(url: url)
        #expect(!config.normTopkProb, "qwen3_next must not renormalize unless the config says so")
        #expect(config.deltaLayout == .fusedInterleaved)
    }

    /// An explicit value always wins over the family default.
    @Test func explicitNormTopkProbWins() throws {
        var text = Self.textConfigCommon()
        text["norm_topk_prob"] = false
        let url = try Self.write([
            "model_type": "qwen3_5_moe",
            "text_config": text,
        ])
        #expect(!(try QwenConfig(url: url).normTopkProb))
    }

    @Test func rejectsNonpositiveContextWindow() throws {
        var text = Self.textConfigCommon()
        text["max_position_embeddings"] = 0
        let url = try Self.write(text)

        #expect(throws: QwenConfig.Error.self) {
            _ = try QwenConfig(url: url)
        }
    }
}
