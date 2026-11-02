import Foundation
import Testing
@testable import SwiftletCore

/// mlx-lm records a quantization *mode* next to `group_size`/`bits`:
/// `affine` (the default), `mxfp4`, `nvfp4` or `mxfp8`. Only `affine` stores
/// the packed uint32 + scales + biases layout `Checkpoint` dequantizes; the
/// others carry e8m0/e4m3 scales and no biases. Before the mode was read, an
/// `mxfp4` checkpoint (`mlx-community/Qwen3.6-35B-A3B-mxfp4` is one) was
/// treated as affine and failed on the missing `.biases` tensor, an error that
/// blamed the file rather than naming the unsupported mode.
@Suite struct CheckpointQuantModeTests {
    static let fixturesDir = MetalModelTests.fixturesDir

    /// Copies the tiny 4-bit fixture and replaces its `quantization` block.
    /// The tensors stay affine; only the declared mode changes, which is
    /// enough because the refusal is decided from `config.json` alone.
    static func checkpoint(quantization: [String: Any]) throws -> URL {
        let src = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let dst = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-mode-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: src, to: dst)
        let cfgURL = dst.appendingPathComponent("config.json")
        var cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: cfgURL)) as! [String: Any]
        cfg["quantization"] = quantization
        try JSONSerialization.data(withJSONObject: cfg).write(to: cfgURL)
        return dst
    }

    static func expectRefused(mode: String, module: String, _ open: () throws -> Void) {
        do {
            try open()
            Issue.record("a \"\(mode)\" checkpoint was opened as affine")
        } catch Checkpoint.Error.unsupportedQuantMode(let m, let mod) {
            #expect(m == mode)
            #expect(mod == module, "the refusal must name the offending module")
        } catch {
            Issue.record("refused with \(error), not by quantization mode")
        }
    }

    /// The shipped fixture predates the field; it must read as affine.
    @Test func absentModeIsAffine() throws {
        let ckpt = try Checkpoint(dir: Self.fixturesDir.appendingPathComponent("tiny-model-q4"))
        #expect(ckpt.defaultQuant?.mode == Checkpoint.affineMode)
        #expect(ckpt.quantSpec(for: "model.embed_tokens")?.mode == Checkpoint.affineMode)
    }

    /// An explicit `affine` mode, at the top level and on a per-module
    /// override, dequantizes exactly as the mode-less fixture does.
    @Test func explicitAffineModeLoadsIdentically() throws {
        let dir = try Self.checkpoint(quantization: [
            "group_size": 32, "bits": 4, "mode": Checkpoint.affineMode,
            "model.layers.0.mlp.gate": ["group_size": 32, "bits": 4, "mode": Checkpoint.affineMode],
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        let explicit = try Checkpoint(dir: dir)
        let original = try Checkpoint(dir: Self.fixturesDir.appendingPathComponent("tiny-model-q4"))
        #expect(explicit.quantSpec(for: "model.layers.0.mlp.gate")?.mode == Checkpoint.affineMode)
        for module in ["model.embed_tokens", "model.layers.0.mlp.gate"] {
            #expect(try explicit.moduleWeight(module).map(\.bitPattern)
                    == original.moduleWeight(module).map(\.bitPattern),
                    "\(module): explicit affine mode changed the dequantized bytes")
        }
    }

    /// A per-module override without `mode` is affine: MLX passes the
    /// override dict to `to_quantized`, whose default is affine, and does not
    /// inherit the checkpoint-level mode.
    @Test func overrideWithoutModeIsAffine() throws {
        let dir = try Self.checkpoint(quantization: [
            "group_size": 32, "bits": 4, "mode": Checkpoint.affineMode,
            "model.layers.0.mlp.gate": ["group_size": 32, "bits": 4],
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        let ckpt = try Checkpoint(dir: dir)
        #expect(ckpt.quantSpec(for: "model.layers.0.mlp.gate")?.mode == Checkpoint.affineMode)
        _ = try ckpt.moduleWeight("model.layers.0.mlp.gate")
    }

    /// A non-affine checkpoint default is refused at open, by name, before
    /// any tensor is read.
    @Test(arguments: ["mxfp4", "nvfp4", "mxfp8"])
    func nonAffineDefaultIsRefusedAtOpen(_ mode: String) throws {
        let dir = try Self.checkpoint(quantization: ["group_size": 32, "bits": 4, "mode": mode])
        defer { try? FileManager.default.removeItem(at: dir) }
        Self.expectRefused(mode: mode, module: "quantization") {
            _ = try Checkpoint(dir: dir)
        }
    }

    /// An affine checkpoint with one non-affine module is refused at open and
    /// the refusal names that module.
    @Test func nonAffineOverrideIsRefusedAtOpen() throws {
        let dir = try Self.checkpoint(quantization: [
            "group_size": 32, "bits": 4, "mode": Checkpoint.affineMode,
            "model.layers.0.mlp.gate": ["group_size": 32, "bits": 4, "mode": "nvfp4"],
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        Self.expectRefused(mode: "nvfp4", module: "model.layers.0.mlp.gate") {
            _ = try Checkpoint(dir: dir)
        }
    }

    /// The repacker is the first thing a user points at a downloaded
    /// checkpoint; it must surface the mode refusal, not a missing tensor.
    @Test func repackRefusesNonAffineCheckpointByMode() throws {
        let src = try Self.checkpoint(quantization: ["group_size": 32, "bits": 4, "mode": "mxfp4"])
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("mode-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: out)
        }
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        Self.expectRefused(mode: "mxfp4", module: "quantization") {
            try repacker.repack()
        }
        #expect(!FileManager.default.fileExists(atPath: out.path), "refusal must not leave a partial container")
    }

    /// The streaming installer reads `config.json` before planning the shard
    /// download; it must refuse a non-affine checkpoint there, before the
    /// first weight byte and before it writes anything into the container.
    /// The local-directory source runs the same code as the HTTP sources.
    @Test func streamingInstallRefusesNonAffineBeforeDownloading() throws {
        let src = try Self.checkpoint(quantization: ["group_size": 32, "bits": 4, "mode": "mxfp4"])
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("mode-stream-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: out)
        }
        let installer = StreamingInstaller(source: .localDirectory(src), outputDir: out)
        installer.log = { _ in }
        Self.expectRefused(mode: "mxfp4", module: "quantization") {
            try installer.install()
        }
        for file in ["config.json", "manifest.json", "packed_experts/layer_00.bin", "model.safetensors"] {
            #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent(file).path),
                    "\(file) must not be written for a refused checkpoint")
        }
    }

    // MARK: mlx-lm `3696eb2` (nvfp4 global scale) and `05b7aed` (prism_hadamard_qwen35)

    /// Copies the tiny 4-bit fixture and appends one tensor to its
    /// `model.safetensors`. The existing tensors keep their bytes, so the
    /// checkpoint is a valid affine one plus the tensor under test.
    static func checkpoint(extraTensor name: String, dtype: String = "F32", shape: [Int]) throws -> URL {
        let dst = try checkpoint(quantization: ["group_size": 32, "bits": 4, "mode": Checkpoint.affineMode])
        let url = dst.appendingPathComponent("model.safetensors")
        let file = try SafetensorsFile(url: url)
        var tensors: [(name: String, dtype: String, shape: [Int], bytes: Data)] = []
        for tensorName in file.tensors.keys.sorted() {
            let (info, bytes) = try file.raw(tensorName)
            tensors.append((tensorName, info.dtype, info.shape, bytes))
        }
        let count = shape.reduce(1, *) * SafetensorsFile.bytesPerElement(dtype)!
        tensors.append((name, dtype, shape, Data(repeating: 0, count: count)))
        try SafetensorsFile.write(to: url, tensors: tensors)
        return dst
    }

    static func expectRefused(tensor key: String, module: String, _ open: () throws -> Void) {
        do {
            try open()
            Issue.record("a checkpoint carrying \(key) was opened as plain affine")
        } catch Checkpoint.Error.unexpectedQuantTensor(let k, let m, let meaning) {
            #expect(k == key, "the refusal must name the tensor")
            #expect(m == module, "the refusal must name the module the tensor belongs to")
            #expect(!meaning.isEmpty)
            let text = Checkpoint.Error.unexpectedQuantTensor(key: k, module: m, meaning: meaning).description
            #expect(text.contains("\"\(key)\"") && text.contains(module))
        } catch {
            Issue.record("refused with \(error), not by the unexpected quantization tensor")
        }
    }

    /// mlx-lm `3696eb2` records an NVIDIA ModelOpt NVFP4 import as
    /// `{"group_size": 16, "bits": 4, "mode": "nvfp4"}` and, on a switch layer
    /// that holds a per-expert tensor scale, a per-module entry with
    /// `"global_scale": true` next to the mode. Both spellings are the mode
    /// `nvfp4` and are refused by that name, at the top level and per module.
    @Test func nvfp4AsMlxLmRecordsItIsRefusedByName() throws {
        let top = try Self.checkpoint(quantization: ["group_size": 16, "bits": 4, "mode": "nvfp4"])
        defer { try? FileManager.default.removeItem(at: top) }
        Self.expectRefused(mode: "nvfp4", module: "quantization") { _ = try Checkpoint(dir: top) }

        let perModule = try Self.checkpoint(quantization: [
            "group_size": 64, "bits": 4, "mode": Checkpoint.affineMode,
            "model.layers.0.mlp.switch_mlp": [
                "group_size": 16, "bits": 4, "mode": "nvfp4", "global_scale": true,
            ],
        ])
        defer { try? FileManager.default.removeItem(at: perModule) }
        Self.expectRefused(mode: "nvfp4", module: "model.layers.0.mlp.switch_mlp") {
            _ = try Checkpoint(dir: perModule)
        }
        let text = Checkpoint.Error.unsupportedQuantMode(mode: "nvfp4", module: "quantization").description
        #expect(text.contains("\"nvfp4\""), "the message must spell the mode")
    }

    /// mlx-lm `05b7aed` ships Hadamard-rotated 2-bit Qwen3.5 as a model type,
    /// `prism_hadamard_qwen35`, not as a quantization mode: its config declares
    /// `{"bits": 2, "group_size": 128, "mode": "affine"}` and the rotation
    /// lives in per-module `signs` tensors. The family check (#39) refuses it
    /// by that name at the repacker and the installer before any shard is
    /// read, so the mode check never has to see it.
    @Test func hadamardPackIsRefusedByItsModelTypeName() throws {
        let dir = try Self.checkpoint(quantization: ["bits": 2, "group_size": 128, "mode": Checkpoint.affineMode])
        let cfgURL = dir.appendingPathComponent("config.json")
        var cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: cfgURL)) as! [String: Any]
        cfg["model_type"] = "prism_hadamard_qwen35"
        cfg["schema_version"] = 1
        cfg["tensor_namespace"] = "mlx-lm-text"
        try JSONSerialization.data(withJSONObject: cfg).write(to: cfgURL)
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("hadamard-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: dir)
            try? FileManager.default.removeItem(at: out)
        }

        func expectRefusedByFamily(_ open: () throws -> Void) {
            do {
                try open()
                Issue.record("a prism_hadamard_qwen35 pack was accepted")
            } catch QwenConfig.Error.unsupportedModelType(let type) {
                #expect(type == "prism_hadamard_qwen35")
                #expect(QwenConfig.Error.unsupportedModelType(type).description.contains("\"prism_hadamard_qwen35\""))
            } catch {
                Issue.record("refused with \(error), not by model_type")
            }
        }
        var repacker = QpackRepacker(checkpointDir: dir, outputDir: out)
        repacker.log = { _ in }
        expectRefusedByFamily { try repacker.repack() }
        #expect(!FileManager.default.fileExists(atPath: out.path))
        let installer = StreamingInstaller(source: .localDirectory(dir), outputDir: out)
        installer.log = { _ in }
        expectRefusedByFamily { try installer.install() }
        #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent("config.json").path))
    }

    /// An affine checkpoint whose expert module carries an nvfp4
    /// `global_scale` tensor is refused at open, naming the tensor and the
    /// module. Before this it opened as plain affine: the scale was never
    /// read, and a repack copied the module without it.
    @Test func affineCheckpointWithGlobalScaleIsRefusedAtOpen() throws {
        let module = "model.layers.0.mlp.switch_mlp.gate_proj"
        let dir = try Self.checkpoint(extraTensor: module + ".global_scale", shape: [8])
        defer { try? FileManager.default.removeItem(at: dir) }
        Self.expectRefused(tensor: module + ".global_scale", module: module) {
            _ = try Checkpoint(dir: dir)
        }
    }

    /// A Hadamard pack whose `model_type` was edited back to a Qwen family
    /// still carries its `signs` vectors; the open refuses those by name
    /// rather than dequantizing rotated words as if they were not.
    @Test func affineCheckpointWithHadamardSignsIsRefusedAtOpen() throws {
        let module = "model.layers.3.self_attn.q_proj"
        let dir = try Self.checkpoint(extraTensor: module + ".signs", shape: [64])
        defer { try? FileManager.default.removeItem(at: dir) }
        Self.expectRefused(tensor: module + ".signs", module: module) {
            _ = try Checkpoint(dir: dir)
        }
    }

    /// The repacker opens the checkpoint first, so it refuses the same way and
    /// writes no container.
    @Test func repackRefusesAffineCheckpointWithGlobalScale() throws {
        let module = "model.layers.0.mlp.switch_mlp.gate_proj"
        let src = try Self.checkpoint(extraTensor: module + ".global_scale", shape: [8])
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("gscale-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: out)
        }
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        Self.expectRefused(tensor: module + ".global_scale", module: module) {
            try repacker.repack()
        }
        #expect(!FileManager.default.fileExists(atPath: out.path), "refusal must not leave a partial container")
    }

    /// The streaming installer only sees the shard headers before it streams;
    /// it must refuse from those, before the first weight byte, so nothing of
    /// the expert blobs, the dense file or the manifest is written.
    @Test func streamingInstallRefusesGlobalScaleBeforeTheFirstWeightByte() throws {
        let module = "model.layers.0.mlp.switch_mlp.gate_proj"
        let src = try Self.checkpoint(extraTensor: module + ".global_scale", shape: [8])
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("gscale-stream-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: out)
        }
        let installer = StreamingInstaller(source: .localDirectory(src), outputDir: out)
        installer.log = { _ in }
        Self.expectRefused(tensor: module + ".global_scale", module: module) {
            try installer.install()
        }
        for file in ["manifest.json", "packed_experts/layer_00.bin", "packed_experts/layout.json", "model.safetensors"] {
            #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent(file).path),
                    "\(file) must not be written for a refused checkpoint")
        }
    }

    /// The deciding function alone: the first offending tensor in name order
    /// is reported, the module is the name minus the suffix, and the ordinary
    /// affine tensors (`weight`, `scales`, `biases`) and plain tensors pass.
    @Test func foreignQuantTensorCheckNamesTheFirstOffender() throws {
        try Checkpoint.checkQuantTensors(names: [
            "model.embed_tokens.weight", "model.embed_tokens.scales", "model.embed_tokens.biases",
            "model.layers.0.input_layernorm.weight", "model.layers.0.linear_attn.A_log",
        ])
        do {
            try Checkpoint.checkQuantTensors(names: [
                "model.layers.1.mlp.switch_mlp.up_proj.global_scale",
                "model.layers.0.mlp.switch_mlp.gate_proj.global_scale",
                "model.layers.0.mlp.switch_mlp.gate_proj.weight",
            ])
            Issue.record("global_scale tensors were accepted")
        } catch Checkpoint.Error.unexpectedQuantTensor(let key, let module, _) {
            #expect(key == "model.layers.0.mlp.switch_mlp.gate_proj.global_scale")
            #expect(module == "model.layers.0.mlp.switch_mlp.gate_proj")
        }
        #expect(Checkpoint.foreignQuantTensors.map(\.suffix) == ["global_scale", "signs"])
    }
}
