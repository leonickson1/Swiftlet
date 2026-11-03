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

    // MARK: Hugging Face / NVIDIA ModelOpt quantization metadata (mlx-lm `3696eb2`, `load_model`)

    /// Copies the tiny 4-bit fixture, drops mlx-lm's `quantization` block
    /// unless `keepMLXBlock`, merges `config` into `config.json`, and writes
    /// `sidecar` as `hf_quant_config.json` beside it when given. The tensors
    /// stay the fixture's affine ones: the refusal is decided from the
    /// metadata alone, as mlx-lm's load is.
    static func checkpoint(hfConfig config: [String: Any], sidecar: [String: Any]? = nil,
                           keepMLXBlock: Bool = false) throws -> URL {
        let src = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let dst = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-hfquant-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: src, to: dst)
        let cfgURL = dst.appendingPathComponent("config.json")
        var cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: cfgURL)) as! [String: Any]
        if !keepMLXBlock { cfg.removeValue(forKey: "quantization") }
        for (key, value) in config { cfg[key] = value }
        try JSONSerialization.data(withJSONObject: cfg).write(to: cfgURL)
        if let sidecar {
            try JSONSerialization.data(withJSONObject: sidecar)
                .write(to: dst.appendingPathComponent("hf_quant_config.json"))
        }
        return dst
    }

    /// `quantization_config` as transformers writes it for an fp8 checkpoint
    /// (`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`, a family this engine accepts).
    static var fp8QuantizationConfig: [String: Any] { [
        "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
        "modules_to_not_convert": ["lm_head", "model.layers.0.mlp.gate"],
    ] }

    /// `hf_quant_config.json` as NVIDIA ModelOpt writes it
    /// (`nvidia/Qwen3-235B-A22B-FP4`, whose `config.json` carries no
    /// `quantization_config` at all): the algorithm sits under
    /// `quantization.quant_algo`; there is no `quant_method`.
    static var modelOptSidecar: [String: Any] { [
        "producer": ["name": "modelopt", "version": "0.33.0"],
        "quantization": ["quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8", "group_size": 16,
                         "exclude_modules": ["model.layers.0.mlp.gate"]],
    ] }

    static func expectRefused(hfSource source: String, algorithm: String, _ open: () throws -> Void) {
        do {
            try open()
            Issue.record("a checkpoint quantized by \(source) (\(algorithm)) was opened as plain weights")
        } catch Checkpoint.Error.hfQuantizedCheckpoint(let s, let a) {
            #expect(s == source, "the refusal must name the file and key the metadata came from")
            #expect(a == algorithm, "the refusal must name the algorithm the metadata declares")
            let text = Checkpoint.Error.hfQuantizedCheckpoint(source: s, algorithm: a).description
            #expect(text.contains(source) && text.contains(algorithm) && text.contains("mlx_lm.convert"),
                    "the message must name the source, the algorithm and the converter")
        } catch {
            Issue.record("refused with \(error), not by the Hugging Face quantization metadata")
        }
    }

    /// A transformers `quantization_config` with no mlx-lm `quantization`
    /// block is refused at open, naming the key and its `quant_method`.
    /// Before this the directory opened as unquantized (`defaultQuant` nil):
    /// its packed tensors still answered `isQuantized`, so the first weight
    /// read failed with `missing tensor model.embed_tokens.scales`, a tensor
    /// the file contains, and an fp8 checkpoint in Hugging Face layout read
    /// as plain bf16.
    @Test func hfQuantizationConfigIsRefusedAtOpen() throws {
        let dir = try Self.checkpoint(hfConfig: ["quantization_config": Self.fp8QuantizationConfig])
        defer { try? FileManager.default.removeItem(at: dir) }
        Self.expectRefused(hfSource: "config.json quantization_config", algorithm: "fp8") {
            _ = try Checkpoint(dir: dir)
        }
    }

    /// A multimodal checkpoint nests the block under `text_config`; mlx-lm
    /// reads it there when the top level has none, and so does the refusal.
    @Test func textConfigQuantizationConfigIsRefusedAtOpen() throws {
        let dir = try Self.checkpoint(hfConfig: [
            "text_config": ["model_type": "qwen3_next",
                            "quantization_config": ["quant_method": "compressed-tensors", "format": "float-quantized"]],
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        Self.expectRefused(hfSource: "config.json text_config.quantization_config", algorithm: "compressed-tensors") {
            _ = try Checkpoint(dir: dir)
        }
    }

    /// A ModelOpt export declares nothing in `config.json`; the sidecar alone
    /// is refused, naming the file and its `quant_algo`.
    @Test func modelOptSidecarIsRefusedAtOpen() throws {
        let dir = try Self.checkpoint(hfConfig: [:], sidecar: Self.modelOptSidecar)
        defer { try? FileManager.default.removeItem(at: dir) }
        Self.expectRefused(hfSource: "hf_quant_config.json", algorithm: "NVFP4") {
            _ = try Checkpoint(dir: dir)
        }
    }

    /// mlx-lm writes `quantization_config` as a copy of its own `quantization`
    /// block on every quantized convert (`save_config`), so the key beside the
    /// block means "converted by mlx-lm", not "quantized by someone else". A
    /// checkpoint carrying the block plus every HF source opens and
    /// dequantizes to the fixture's exact bytes.
    @Test func mlxLmBlockBesideHFMetadataIsNotRefused() throws {
        let dir = try Self.checkpoint(hfConfig: [
            "quantization_config": ["group_size": 32, "bits": 4],
            "text_config": ["quantization_config": Self.fp8QuantizationConfig],
        ], sidecar: Self.modelOptSidecar, keepMLXBlock: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let converted = try Checkpoint(dir: dir)
        let original = try Checkpoint(dir: Self.fixturesDir.appendingPathComponent("tiny-model-q4"))
        #expect(converted.defaultQuant?.bits == 4 && converted.defaultQuant?.groupSize == 32)
        for module in ["model.embed_tokens", "model.layers.0.mlp.gate"] {
            #expect(try converted.moduleWeight(module).map(\.bitPattern)
                    == original.moduleWeight(module).map(\.bitPattern),
                    "\(module): HF metadata beside the mlx-lm block changed the dequantized bytes")
        }
    }

    /// The repacker opens the checkpoint first, so it refuses the same way and
    /// writes no container.
    @Test func repackRefusesHFQuantizedCheckpointByName() throws {
        let src = try Self.checkpoint(hfConfig: ["quantization_config": Self.fp8QuantizationConfig])
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("hfquant-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: out)
        }
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        Self.expectRefused(hfSource: "config.json quantization_config", algorithm: "fp8") {
            try repacker.repack()
        }
        #expect(!FileManager.default.fileExists(atPath: out.path), "refusal must not leave a partial container")
    }

    /// The streaming installer decides from `config.json` and the sidecar it
    /// fetches beside it, before its shard plan and before it writes anything
    /// into the output directory, so a refused checkpoint costs no weight
    /// byte. All three sources, each alone.
    @Test func streamingInstallRefusesHFQuantizedBeforeTheFirstByte() throws {
        let cases: [(config: [String: Any], sidecar: [String: Any]?, source: String, algorithm: String)] = [
            (["quantization_config": Self.fp8QuantizationConfig], nil, "config.json quantization_config", "fp8"),
            (["text_config": ["quantization_config": ["quant_method": "awq", "bits": 4]]], nil,
             "config.json text_config.quantization_config", "awq"),
            ([:], Self.modelOptSidecar, "hf_quant_config.json", "NVFP4"),
        ]
        for c in cases {
            let src = try Self.checkpoint(hfConfig: c.config, sidecar: c.sidecar)
            let out = FileManager.default.temporaryDirectory
                .appendingPathComponent("hfquant-stream-\(UUID().uuidString).qpack")
            defer {
                try? FileManager.default.removeItem(at: src)
                try? FileManager.default.removeItem(at: out)
            }
            let installer = StreamingInstaller(source: .localDirectory(src), outputDir: out)
            installer.log = { _ in }
            Self.expectRefused(hfSource: c.source, algorithm: c.algorithm) { try installer.install() }
            for file in ["config.json", "manifest.json", "packed_experts/layer_00.bin",
                         "packed_experts/layout.json", "model.safetensors"] {
                #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent(file).path),
                        "\(file) must not be written for a checkpoint refused by \(c.source)")
            }
        }
    }

    /// An mlx-lm convert that still carries HF metadata beside its
    /// `quantization` block streams as before, and the sidecar is not fetched
    /// for it.
    @Test func streamingInstallAcceptsMLXConvertCarryingHFMetadata() throws {
        let src = try Self.checkpoint(hfConfig: ["quantization_config": ["group_size": 32, "bits": 4]],
                                      sidecar: Self.modelOptSidecar, keepMLXBlock: true)
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("hfquant-ok-\(UUID().uuidString).qpack")
        defer {
            try? FileManager.default.removeItem(at: src)
            try? FileManager.default.removeItem(at: out)
        }
        let installer = StreamingInstaller(source: .localDirectory(src), outputDir: out)
        installer.log = { _ in }
        try installer.install()
        let manifest = try JSONDecoder().decode(
            Qpack.Manifest.self, from: Data(contentsOf: out.appendingPathComponent("manifest.json")))
        #expect(manifest.quantBits == 4 && manifest.quantGroupSize == 32)
    }

    /// The deciding function alone: mlx-lm's order of the three sources, the
    /// algorithm each shape names, the sidecar read only when `config.json`
    /// names nothing, and the mlx-lm block switching the whole check off.
    @Test func hfQuantizationCheckNamesTheSourceAndAlgorithm() throws {
        var sidecarReads = 0
        func sidecar(_ value: [String: Any]?) -> () throws -> [String: Any]? {
            { sidecarReads += 1; return value }
        }
        let embeddedModelOpt: [String: Any] = ["quant_method": "modelopt", "quantization": ["quant_algo": "NVFP4"]]

        // Order: the top level, then text_config, then the sidecar.
        var found = try Checkpoint.hfQuantization(fromConfig: [
            "quantization_config": Self.fp8QuantizationConfig,
            "text_config": ["quantization_config": ["quant_method": "awq"]],
        ], hfQuantConfig: sidecar(Self.modelOptSidecar))
        #expect(found?.source == "config.json quantization_config" && found?.algorithm == "fp8")
        found = try Checkpoint.hfQuantization(fromConfig: ["text_config": ["quantization_config": embeddedModelOpt]],
                                              hfQuantConfig: sidecar(Self.modelOptSidecar))
        #expect(found?.source == "config.json text_config.quantization_config" && found?.algorithm == "modelopt/NVFP4")
        #expect(sidecarReads == 0, "the sidecar is read only when config.json names nothing")
        found = try Checkpoint.hfQuantization(fromConfig: [:], hfQuantConfig: sidecar(Self.modelOptSidecar))
        #expect(found?.source == "hf_quant_config.json" && found?.algorithm == "NVFP4")
        #expect(sidecarReads == 1)
        #expect(try Checkpoint.hfQuantization(fromConfig: [:], hfQuantConfig: sidecar(nil)) == nil)

        // Algorithm spellings: quant_algo at the top level, neither key, an empty object.
        #expect(try Checkpoint.hfQuantization(fromConfig: ["quantization_config": ["quant_algo": "FP8"]],
                                              hfQuantConfig: sidecar(nil))?.algorithm == "FP8")
        #expect(try Checkpoint.hfQuantization(fromConfig: ["quantization_config": ["bits": 8]],
                                              hfQuantConfig: sidecar(nil))?.algorithm == "unknown")
        #expect(try Checkpoint.hfQuantization(fromConfig: ["quantization_config": [String: Any]()],
                                              hfQuantConfig: sidecar(nil)) == nil,
                "an empty quantization_config declares nothing, as mlx-lm treats it")
        // A present but unreadable sidecar still counts; an absent one is nil.
        #expect(Checkpoint.hfQuantConfig(from: Data("not json".utf8))?.isEmpty == true)
        #expect(Checkpoint.hfQuantConfig(from: nil) == nil)

        // The mlx-lm block switches the check off whatever else is present, without reading the sidecar.
        sidecarReads = 0
        try Checkpoint.checkHFQuantization(config: [
            "quantization": ["group_size": 32, "bits": 4],
            "quantization_config": Self.fp8QuantizationConfig,
        ], hfQuantConfig: sidecar(Self.modelOptSidecar))
        #expect(sidecarReads == 0)
        // Without it, the same config is refused by name.
        Self.expectRefused(hfSource: "config.json quantization_config", algorithm: "fp8") {
            try Checkpoint.checkHFQuantization(config: ["quantization_config": Self.fp8QuantizationConfig],
                                               hfQuantConfig: sidecar(nil))
        }
    }
}
