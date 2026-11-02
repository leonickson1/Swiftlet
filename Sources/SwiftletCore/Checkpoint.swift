import Foundation

/// Uniform reader over an mlx-lm checkpoint directory: single `model.safetensors`
/// or sharded `model-XXXXX-of-XXXXX.safetensors` + `model.safetensors.index.json`,
/// with transparent MLX affine dequantization (packed uint32 weight + scales +
/// biases per group) for 2/4/8-bit tensors.
public final class Checkpoint {
    /// mlx-lm's default `quantization.mode`, and the only packed layout this
    /// reader dequantizes: uint32 words + scales + biases per group. The other
    /// modes (`mxfp4`, `nvfp4`, `mxfp8`) store e8m0/e4m3 scales and no biases.
    public static let affineMode = "affine"

    public struct QuantSpec: Sendable {
        public let groupSize: Int
        public let bits: Int
        /// `quantization.mode` as mlx-lm recorded it. Absent from the config
        /// means `affine`: that is mlx-lm's default, and checkpoints converted
        /// before the field existed omit it.
        public let mode: String

        public init(groupSize: Int, bits: Int, mode: String = Checkpoint.affineMode) {
            self.groupSize = groupSize
            self.bits = bits
            self.mode = mode
        }
    }

    public enum Error: Swift.Error, CustomStringConvertible {
        case missingTensor(String)
        case unsupportedBits(Int)
        case badShape(String)
        /// The checkpoint declares an MLX quantization mode other than `affine`
        /// for `module` (`"quantization"` when it is the checkpoint default).
        /// Its packed layout cannot be dequantized as affine, so the checkpoint
        /// is refused before any tensor is read.
        case unsupportedQuantMode(mode: String, module: String)
        /// A module declared affine carries a quantization tensor that affine
        /// dequantization has no use for: `key` is the full tensor name,
        /// `module` the module it belongs to, `meaning` what the tensor is
        /// (`Checkpoint.foreignQuantTensors`). Reading the module as plain
        /// affine would drop the tensor and produce wrong weights with no
        /// error, so the checkpoint is refused before any weight byte is read.
        case unexpectedQuantTensor(key: String, module: String, meaning: String)
        /// `dir` has no `config.json`, so it is not a checkpoint directory (nor
        /// a qpack container, which carries a copy). Refused at open: before
        /// this the absent file read as an empty config, the directory opened
        /// as an unquantized checkpoint, and the failure surfaced later as a
        /// missing `.scales` tensor that blamed the file.
        case missingConfig(String)
        /// `config.json` exists but cannot be read as a JSON object.
        case malformedConfig(path: String, reason: String)

        public var description: String {
            switch self {
            case .missingTensor(let name): return "missing tensor \(name)"
            case .missingConfig(let dir):
                return "no config.json in \(dir): not a checkpoint directory "
                    + "(a checkpoint carries config.json, model.safetensors or its shards with "
                    + "model.safetensors.index.json, and the tokenizer files; copy them from the source)"
            case .malformedConfig(let path, let reason):
                return "config.json at \(path) is not a JSON object (\(reason))"
            case .unsupportedBits(let bits): return "unsupported \(bits)-bit quantization (affine 4- and 8-bit only)"
            case .badShape(let name): return "bad shape for tensor \(name)"
            case .unsupportedQuantMode(let mode, let module):
                return "unsupported MLX quantization mode \"\(mode)\" (\(module)); "
                    + "only \"\(Checkpoint.affineMode)\" checkpoints (packed weight + scales + biases) can be read"
            case .unexpectedQuantTensor(let key, let module, let meaning):
                return "unexpected quantization tensor \"\(key)\" beside the affine weights of \(module): "
                    + "\(meaning); dequantizing the module as plain affine would silently drop it, "
                    + "and only \"\(Checkpoint.affineMode)\" checkpoints (packed weight + scales + biases) can be read"
            }
        }
    }

    public let dir: URL
    public let defaultQuant: QuantSpec?
    /// Per-module overrides keyed by module path (e.g. "model.layers.0.mlp.gate").
    public let quantOverrides: [String: QuantSpec]

    private var files: [SafetensorsFile] = []
    private var fileURLs: [URL] = []
    private var tensorToFile: [String: Int] = [:]

    /// Parses mlx-lm's quantization block and refuses any mode this reader
    /// cannot dequantize, so every entry point (checkpoint open, repack, the
    /// streaming installer before its first weight byte) decides the same way.
    ///
    /// Config shape: {"quantization": {"group_size": 64, "bits": 4, "mode": "affine",
    /// "model.layers.N.mlp.gate": {"group_size": 64, "bits": 8, "mode": "affine"}, ...}}.
    /// A per-module override without `mode` is affine regardless of the
    /// checkpoint default: MLX hands the override dict straight to
    /// `to_quantized`, whose own default is affine, rather than inheriting
    /// the top-level mode (mlx-community's mxfp4 Qwen3.6 builds keep their
    /// routers affine exactly this way). The mode is refused before any
    /// tensor is read: a non-affine checkpoint has no `.biases`, and reading
    /// it as affine failed on a missing tensor that blamed the file.
    public static func quantization(fromConfig cfg: [String: Any]) throws
        -> (default: QuantSpec?, overrides: [String: QuantSpec]) {
        guard let q = cfg["quantization"] as? [String: Any] else { return (nil, [:]) }
        var defQuant: QuantSpec? = nil
        var overrides: [String: QuantSpec] = [:]
        if let g = q["group_size"] as? Int, let b = q["bits"] as? Int {
            defQuant = QuantSpec(groupSize: g, bits: b,
                                 mode: q["mode"] as? String ?? affineMode)
        }
        for (key, value) in q {
            if let sub = value as? [String: Any],
               let g = sub["group_size"] as? Int, let b = sub["bits"] as? Int {
                overrides[key] = QuantSpec(groupSize: g, bits: b,
                                           mode: sub["mode"] as? String ?? affineMode)
            }
        }
        if let d = defQuant, d.mode != affineMode {
            throw Error.unsupportedQuantMode(mode: d.mode, module: "quantization")
        }
        for (module, spec) in overrides.sorted(by: { $0.key < $1.key }) where spec.mode != affineMode {
            throw Error.unsupportedQuantMode(mode: spec.mode, module: module)
        }
        return (defQuant, overrides)
    }

    /// Quantization tensors that are not part of the affine layout (weight +
    /// scales + biases), keyed by the suffix mlx-lm stores them under, with
    /// what each one is. Every mode other than affine is already refused by
    /// name from `config.json`; these catch a checkpoint whose config says
    /// affine while its weights carry a tensor only another layout uses, which
    /// `quantization(fromConfig:)` cannot see. A repacker or installer that
    /// copied the module as plain affine would drop the tensor and ship wrong
    /// weights without an error, so the tensor is refused by name instead.
    ///
    /// - `global_scale`: mlx-lm's nvfp4 per-expert tensor scale on switch
    ///   layers (mlx-lm `3696eb2`, "Add global scale for nvfp4 quantized
    ///   MoEs"): `mx.quantize(..., mode="nvfp4", global_scale=...)` stores a
    ///   `<module>.global_scale` tensor next to `.weight`/`.scales`.
    /// - `signs`: the sign vector of a Hadamard rotation folded into a packed
    ///   module's input axis (mlx-lm `05b7aed`, `prism_hadamard_qwen35`): the
    ///   2-bit affine words are only meaningful after `mx.hadamard_transform`
    ///   of the activations with `<module>.signs`.
    public static let foreignQuantTensors: [(suffix: String, meaning: String)] = [
        ("global_scale", "an nvfp4 per-expert global scale (mlx-lm `mode: nvfp4` with `global_scale`)"),
        ("signs", "a Hadamard rotation sign vector (mlx-lm `prism_hadamard_qwen35`, Hadamard-rotated 2-bit)"),
    ]

    /// Refuses, by tensor and module name, the first tensor in `names` that is
    /// one of `foreignQuantTensors`. Decided from the safetensors headers
    /// alone: `init` runs it after mapping the shard headers and before any
    /// tensor is read, and the streaming installer runs it over the headers it
    /// has fetched before the first weight byte, so both producers refuse the
    /// same checkpoint the same way.
    public static func checkQuantTensors<S: Sequence>(names: S) throws where S.Element == String {
        var found: [(key: String, module: String, meaning: String)] = []
        for name in names {
            for entry in foreignQuantTensors where name.hasSuffix("." + entry.suffix) {
                let module = String(name.dropLast(entry.suffix.count + 1))
                found.append((name, module, entry.meaning))
            }
        }
        // Deterministic: the same checkpoint names the same tensor every time.
        if let first = found.min(by: { $0.key < $1.key }) {
            throw Error.unexpectedQuantTensor(key: first.key, module: first.module, meaning: first.meaning)
        }
    }

    /// Reads a directory's `config.json` as a JSON object, refusing an absent
    /// or unreadable one by name. `QwenConfig` opens through the same
    /// function, so every opener says the same thing about the same
    /// directory instead of one guessing an empty config and the other
    /// reporting a Foundation file error.
    public static func readConfig(inDirectory dir: URL) throws -> [String: Any] {
        let url = dir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw Error.missingConfig(dir.path)
        }
        let data: Data
        do { data = try Data(contentsOf: url) } catch {
            throw Error.malformedConfig(path: url.path, reason: "cannot read: \(error.localizedDescription)")
        }
        let object: Any
        do { object = try JSONSerialization.jsonObject(with: data) } catch {
            throw Error.malformedConfig(path: url.path, reason: "invalid JSON: \(error.localizedDescription)")
        }
        guard let cfg = object as? [String: Any] else {
            throw Error.malformedConfig(path: url.path, reason: "the top level is not an object")
        }
        return cfg
    }

    public init(dir: URL) throws {
        self.dir = dir

        // Refuses a directory that is not a checkpoint here, by name, before
        // any shard is looked for.
        let cfg = try Self.readConfig(inDirectory: dir)
        // Refuses a non-affine checkpoint here, before any shard is opened.
        let quant = try Self.quantization(fromConfig: cfg)
        defaultQuant = quant.default
        quantOverrides = quant.overrides

        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path),
           let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any],
           let weightMap = index["weight_map"] as? [String: String] {
            var fileIndex: [String: Int] = [:]
            for (tensor, fileName) in weightMap {
                if fileIndex[fileName] == nil {
                    fileIndex[fileName] = files.count
                    let url = dir.appendingPathComponent(fileName)
                    files.append(try SafetensorsFile(url: url))
                    fileURLs.append(url)
                }
                tensorToFile[tensor] = fileIndex[fileName]!
            }
        } else {
            let url = dir.appendingPathComponent("model.safetensors")
            files.append(try SafetensorsFile(url: url))
            fileURLs.append(url)
            for name in files[0].tensors.keys { tensorToFile[name] = 0 }
        }
        // The headers are mapped and no tensor has been read yet: refuse a
        // quantization tensor the affine layout has no place for, by name,
        // before a caller dequantizes the module around it.
        try Self.checkQuantTensors(names: tensorToFile.keys)
    }

    /// Multimodal checkpoints (qwen3_5_moe) prefix text weights with
    /// "language_model."; callers use unprefixed names and we resolve.
    private func resolve(_ name: String) -> String {
        if tensorToFile[name] != nil { return name }
        let prefixed = "language_model." + name
        return tensorToFile[prefixed] != nil ? prefixed : name
    }

    public func contains(_ name: String) -> Bool { tensorToFile[resolve(name)] != nil }

    public var tensorNames: [String] { Array(tensorToFile.keys) }

    /// Raw stored bytes + info, no conversion (for repacking).
    public func rawTensor(_ name: String) throws -> (info: SafetensorsFile.TensorInfo, bytes: Data) {
        let r = resolve(name)
        return try file(for: r).raw(r)
    }

    /// Zero-copy tensor byte access (view into the mapped shard).
    public func withRawTensor<T>(_ name: String, _ body: (SafetensorsFile.TensorInfo, UnsafeRawBufferPointer) throws -> T) throws -> T {
        let r = resolve(name)
        return try file(for: r).withRawBytes(r, body)
    }

    /// Shard file URL + byte offset within it (for mmap/GPU zero-copy binding).
    public func tensorLocation(_ name: String) throws -> (url: URL, byteOffset: Int, info: SafetensorsFile.TensorInfo) {
        let r = resolve(name)
        guard let i = tensorToFile[r] else { throw Error.missingTensor(name) }
        let f = files[i]
        return (fileURLs[i], try f.absoluteOffset(r), try f.info(r))
    }

    private func file(for name: String) throws -> SafetensorsFile {
        guard let i = tensorToFile[name] else { throw Error.missingTensor(name) }
        return files[i]
    }

    /// Resolved (possibly prefixed) name plus the shard holding it.
    private func fileAndName(_ name: String) throws -> (SafetensorsFile, String) {
        let r = resolve(name)
        return (try file(for: r), r)
    }

    public func shape(_ name: String) throws -> [Int] {
        let (f, n) = try fileAndName(name)
        return try f.info(n).shape
    }

    /// True when `path.weight` is stored quantized (has companion scales/biases).
    public func isQuantized(_ path: String) -> Bool {
        contains(path + ".scales")
    }

    public func quantSpec(for path: String) -> QuantSpec? {
        guard isQuantized(path) else { return nil }
        return Self.quantSpec(for: path, default: defaultQuant, overrides: quantOverrides)
    }

    /// The spec a quantized module resolves to under a parsed quantization
    /// block: the most specific override suffix (config keys omit shard
    /// prefixes), else the default. `quantSpec(for:)` applies it to an opened
    /// checkpoint; the streaming installer applies it to the config it has
    /// fetched before any shard is read.
    public static func quantSpec(for path: String, default defaultQuant: QuantSpec?,
                                 overrides: [String: QuantSpec]) -> QuantSpec? {
        for (key, spec) in overrides where path.hasSuffix(key) || key.hasSuffix(path) {
            return spec
        }
        return defaultQuant
    }

    /// Dequantized (or plain) weights for a linear/embedding module `path`,
    /// returned row-major with the checkpoint's logical shape.
    public func moduleWeight(_ path: String) throws -> [Float] {
        if isQuantized(path) {
            return try dequantized(path, rowRange: nil)
        }
        let (f, n) = try fileAndName(path + ".weight")
        return try f.floats(n)
    }

    /// A contiguous slice of rows, where a "row" is one vector along the LAST
    /// axis and all leading axes are flattened (matches the quantized layout,
    /// so stacked experts slice as `expert * innerRows ..< (expert+1) * innerRows`).
    public func moduleWeightSlice(_ path: String, rowRange: Range<Int>) throws -> [Float] {
        if isQuantized(path) {
            return try dequantized(path, rowRange: rowRange)
        }
        let (f, wName) = try fileAndName(path + ".weight")
        let rowLen = try f.info(wName).shape.last!
        return try f.floats(wName, elementRange: rowRange.lowerBound * rowLen..<rowRange.upperBound * rowLen)
    }

    /// Plain (never-quantized) tensor: norms, A_log, dt_bias, conv1d, etc.
    public func tensor(_ name: String) throws -> [Float] {
        let (f, n) = try fileAndName(name)
        return try f.floats(n)
    }

    // MARK: - MLX affine dequantization

    /// weight: uint32-packed along the last axis (8x4-bit or 4x8-bit per word),
    /// scales/biases: one per `groupSize` consecutive logical elements.
    /// w[i] = scale[g] * q[i] + bias[g].
    private func dequantized(_ path: String, rowRange: Range<Int>?) throws -> [Float] {
        guard let spec = quantSpec(for: path) else { throw Error.missingTensor(path + ".scales") }
        // `init` already refused non-affine configs; this keeps the affine
        // arithmetic below from ever running on another layout.
        guard spec.mode == Self.affineMode else {
            throw Error.unsupportedQuantMode(mode: spec.mode, module: path)
        }
        guard spec.bits == 4 || spec.bits == 8 else { throw Error.unsupportedBits(spec.bits) }
        let perWord = 32 / spec.bits
        let mask = UInt32((1 << spec.bits) - 1)

        let (f, wName) = try fileAndName(path + ".weight")
        let wInfo = try f.info(wName)
        let shape = wInfo.shape
        guard let packedCols = shape.last else { throw Error.badShape(wName) }
        let logicalCols = packedCols * perWord
        let totalRows = shape.dropLast().reduce(1, *)
        let rows = rowRange ?? 0..<totalRows

        let packed = try f.uint32s(wName, elementRange: rows.lowerBound * packedCols..<rows.upperBound * packedCols)
        let groupsPerRow = logicalCols / spec.groupSize
        // Scales/biases can land in a different shard than the weight when a
        // module straddles a shard boundary — resolve each independently.
        let (sf, sName) = try fileAndName(path + ".scales")
        let (bf, bName) = try fileAndName(path + ".biases")
        let scales = try sf.floats(sName, elementRange: rows.lowerBound * groupsPerRow..<rows.upperBound * groupsPerRow)
        let biases = try bf.floats(bName, elementRange: rows.lowerBound * groupsPerRow..<rows.upperBound * groupsPerRow)

        let rowCount = rows.count
        var out = [Float](repeating: 0, count: rowCount * logicalCols)
        out.withUnsafeMutableBufferPointer { o in
            for r in 0..<rowCount {
                for w in 0..<packedCols {
                    var word = packed[r * packedCols + w]
                    let colBase = w * perWord
                    for j in 0..<perWord {
                        let col = colBase + j
                        let g = col / spec.groupSize
                        let q = Float(word & mask)
                        o[r * logicalCols + col] = scales[r * groupsPerRow + g] * q + biases[r * groupsPerRow + g]
                        word >>= UInt32(spec.bits)
                    }
                }
            }
        }
        return out
    }
}

extension SafetensorsFile {
    /// Raw uint32 words for a packed-quantized tensor, optionally a sub-range
    /// of elements (in words).
    func uint32s(_ name: String, elementRange: Range<Int>? = nil) throws -> [UInt32] {
        let t = try info(name)
        guard t.dtype == "U32" || t.dtype == "I32" else {
            throw Error.unsupportedDtype(t.dtype, tensor: name)
        }
        let r = elementRange ?? 0..<(t.byteRange.count / 4)
        // In-place view of just the requested words: the per-token embedding
        // lookup reads one row of a tensor that is hundreds of MB on a real
        // model, and copying the whole tensor per row was 18% of the M1's
        // decode CPU gap (S3c). memcpy, because the data section is not
        // guaranteed to be word-aligned.
        return withTensorBytes(t, byteRange: (r.lowerBound * 4)..<(r.upperBound * 4)) { raw in
            [UInt32](unsafeUninitializedCapacity: r.count) { out, count in
                if r.count > 0 { memcpy(out.baseAddress!, raw.baseAddress!, r.count * 4) }
                count = r.count
            }
        }
    }

    /// Float conversion over a sub-range of elements (F32/F16/BF16), read
    /// in place (see uint32s).
    func floats(_ name: String, elementRange r: Range<Int>) throws -> [Float] {
        let t = try info(name)
        switch t.dtype {
        case "F32":
            return withTensorBytes(t, byteRange: (r.lowerBound * 4)..<(r.upperBound * 4)) { raw in
                [Float](unsafeUninitializedCapacity: r.count) { out, count in
                    if r.count > 0 { memcpy(out.baseAddress!, raw.baseAddress!, r.count * 4) }
                    count = r.count
                }
            }
        case "F16":
            return withTensorBytes(t, byteRange: (r.lowerBound * 2)..<(r.upperBound * 2)) { raw in
                (0..<r.count).map { Float(raw.loadUnaligned(fromByteOffset: $0 * 2, as: Float16.self)) }
            }
        case "BF16":
            return withTensorBytes(t, byteRange: (r.lowerBound * 2)..<(r.upperBound * 2)) { raw in
                (0..<r.count).map {
                    Float(bitPattern: UInt32(raw.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self)) << 16)
                }
            }
        default:
            throw Error.unsupportedDtype(t.dtype, tensor: name)
        }
    }
}
