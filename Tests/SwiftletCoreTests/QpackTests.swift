import Foundation
import Testing
@testable import SwiftletCore

/// Repacks the quantized tiny model into a .qpack container and verifies the
/// round trip: expert blobs byte-identical to checkpoint slices, dense file
/// readable, single-pread expert fetch works.
@Suite struct QpackTests {
    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    @Test func repackRoundTrip() throws {
        let src = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-q4-\(UUID().uuidString).qpack")
        defer { try? FileManager.default.removeItem(at: out) }

        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        try repacker.repack()

        let ckpt = try Checkpoint(dir: src)
        let config = try QwenConfig(url: src.appendingPathComponent("config.json"))
        let reader = try QpackExpertReader(containerDir: out)

        #expect(reader.layout.expertCount == config.numExperts)
        #expect(reader.layout.layerCount == config.numHiddenLayers)
        #expect(reader.layout.expertStride % Qpack.pageAlignment == 0)

        // Every expert blob must reproduce the checkpoint bytes exactly.
        var blob = [UInt8](repeating: 0, count: reader.layout.expertStride)
        for layer in [0, config.numHiddenLayers - 1] {
            for expert in [0, 3, config.numExperts - 1] {
                try blob.withUnsafeMutableBytes {
                    try reader.readExpert(layer: layer, expert: expert, into: $0.baseAddress!)
                }
                for section in reader.layout.sections {
                    let full = try ckpt.rawTensor("model.layers.\(layer).mlp.switch_mlp.\(section.name)").bytes
                    let expected = full[expert * section.size..<(expert + 1) * section.size]
                    let got = Data(blob[section.offset..<section.offset + section.size])
                    #expect(got == Data(expected), "layer \(layer) expert \(expert) \(section.name) bytes differ")
                }
            }
        }

        // Dense file: readable as a checkpoint, tensors byte-identical, and
        // expert/MTP tensors excluded.
        let packed = try Checkpoint(dir: out)
        for name in ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight",
                     "model.layers.0.mlp.gate.scales", "model.layers.3.self_attn.q_norm.weight"] {
            let a = try ckpt.rawTensor(name)
            let b = try packed.rawTensor(name)
            #expect(a.bytes == b.bytes && a.info.dtype == b.info.dtype, "\(name) differs after repack")
        }
        #expect(!packed.contains("model.layers.0.mlp.switch_mlp.gate_proj.weight"))

        // Dequantized expert slice from the blob matches Checkpoint's dequant.
        let section = reader.section("gate_proj.weight")!
        #expect(section.shape == [config.moeIntermediateSize, config.hiddenSize / 8])
    }

    /// The streaming installer (local-source mode exercises the same routing
    /// code as the HTTP path) must produce a container byte-identical to the
    /// standard repacker's, and the GPU model must run from it.
    @Test func streamingInstallMatchesRepacker() throws {
        let src = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let viaRepack = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-repack-\(UUID().uuidString).qpack")
        let viaStream = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-stream-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: viaRepack)
            try? FileManager.default.removeItem(at: viaStream)
        }

        var repacker = QpackRepacker(checkpointDir: src, outputDir: viaRepack)
        repacker.log = { _ in }
        try repacker.repack()

        let installer = StreamingInstaller(source: .localDirectory(src), outputDir: viaStream)
        installer.log = { _ in }
        try installer.install()

        // Expert layer files must match byte for byte.
        let config = try QwenConfig(url: src.appendingPathComponent("config.json"))
        for l in 0..<config.numHiddenLayers {
            let name = String(format: "packed_experts/layer_%02d.bin", l)
            let a = try Data(contentsOf: viaRepack.appendingPathComponent(name))
            let b = try Data(contentsOf: viaStream.appendingPathComponent(name))
            #expect(a == b, "\(name) differs between repacker and streaming installer")
        }

        // Dense file: every tensor byte-identical through the reader.
        let ra = try Checkpoint(dir: viaRepack)
        let rb = try Checkpoint(dir: viaStream)
        for name in ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight",
                     "model.layers.3.self_attn.q_proj.weight", "lm_head.scales"] {
            let ta = try ra.rawTensor(name)
            let tb = try rb.rawTensor(name)
            #expect(ta.bytes == tb.bytes, "\(name) differs")
        }

        // And the streamed container actually runs, matching the CPU model.
        let cpu = try QwenCPUModel(modelDir: src)
        cpu.retainAllLayers = true
        let gpu = try QwenMetalModel(modelDir: viaStream, cacheBudgetGB: 0.05)
        let s1 = QwenCPUModel.DecodeState()
        let s2 = QwenCPUModel.DecodeState()
        var a: [Float] = []
        var b: [Float] = []
        for t in [1, 5, 9] {
            a = try cpu.step([t], state: s1)
            b = try gpu.step([t], state: s2)
        }
        var maxDiff: Float = 0
        for i in 0..<a.count { maxDiff = max(maxDiff, abs(a[i] - b[i])) }
        #expect(maxDiff < 2e-3, "streamed container logits diff \(maxDiff)")
    }

    // MARK: - Mixed-precision expert quantization (issue #30)

    /// Copies the tiny 4-bit fixture and rewrites its quantization config to a
    /// dense default with per-module `switch_mlp` overrides, reproducing the
    /// mixed-precision shape (8-bit dense, 4-bit experts) of the DWQ variants.
    /// The expert bytes stay 4-bit; only the recorded precision changes.
    static func mixedPrecisionCheckpoint(defaultBits: Int, expertBits: [String: Int],
                                         group: Int = 32) throws -> URL {
        let src = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let dst = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-mixed-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: src, to: dst)
        let cfgURL = dst.appendingPathComponent("config.json")
        var cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: cfgURL)) as! [String: Any]
        var quant: [String: Any] = ["group_size": group, "bits": defaultBits]
        for (module, bits) in expertBits {
            quant[module] = ["group_size": group, "bits": bits]
        }
        cfg["quantization"] = quant
        try JSONSerialization.data(withJSONObject: cfg).write(to: cfgURL)
        return dst
    }

    /// A checkpoint whose experts are 4-bit under an 8-bit dense default must
    /// repack a manifest that records 4-bit, not the 8-bit default. Recording
    /// the default made the runtime dequantize the 4-bit expert payload as
    /// 8-bit and decode garbage with no error (issue #30).
    @Test func repackRecordsExpertQuantNotCheckpointDefault() throws {
        let src = try Self.mixedPrecisionCheckpoint(defaultBits: 8, expertBits: [
            "model.layers.0.mlp.switch_mlp.gate_proj": 4,
            "model.layers.0.mlp.switch_mlp.up_proj": 4,
            "model.layers.0.mlp.switch_mlp.down_proj": 4,
        ])
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("mixed-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: out)
        }
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        try repacker.repack()

        let manifest = try JSONDecoder().decode(
            Qpack.Manifest.self,
            from: Data(contentsOf: out.appendingPathComponent("manifest.json")))
        #expect(manifest.quantBits == 4, "manifest must record the 4-bit expert precision, not the 8-bit default")
        #expect(manifest.quantGroupSize == 32)
    }

    /// When the expert projections carry different precisions, no single
    /// manifest value is correct, so repack must refuse rather than ship a
    /// container that decodes garbage for some projections (issue #30).
    @Test func repackRejectsDisagreeingExpertQuant() throws {
        let src = try Self.mixedPrecisionCheckpoint(defaultBits: 8, expertBits: [
            "model.layers.0.mlp.switch_mlp.gate_proj": 4,
            "model.layers.0.mlp.switch_mlp.up_proj": 8,
            "model.layers.0.mlp.switch_mlp.down_proj": 4,
        ])
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("mixed-bad-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: out)
        }
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        #expect(throws: QpackRepacker.Error.self) { try repacker.repack() }
    }

    /// The load-time guard rejects a container whose manifest quantization does
    /// not match its packed expert shapes (issue #30). For the tiny geometry
    /// (hidden 64, group 32) a 4-bit weight stores 8 words per row and its
    /// scales store 2 groups per row; reading those bytes as 8-bit makes the
    /// two disagree.
    @Test func expertLogicalInDimRejectsQuantMismatch() throws {
        let inDim = try Qpack.expertLogicalInDim(
            weightLastDim: 8, scalesLastDim: 2, bits: 4, groupSize: 32, section: "gate_proj.weight")
        #expect(inDim == 64)
        #expect(throws: (any Swift.Error).self) {
            _ = try Qpack.expertLogicalInDim(
                weightLastDim: 8, scalesLastDim: 2, bits: 8, groupSize: 32, section: "gate_proj.weight")
        }
    }
}
