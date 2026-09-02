import Foundation

/// Uniform reader over an mlx-lm checkpoint directory: single `model.safetensors`
/// or sharded `model-XXXXX-of-XXXXX.safetensors` + `model.safetensors.index.json`,
/// with transparent MLX affine dequantization (packed uint32 weight + scales +
/// biases per group) for 2/4/8-bit tensors.
public final class Checkpoint {
    public struct QuantSpec: Sendable {
        public let groupSize: Int
        public let bits: Int
    }

    public enum Error: Swift.Error {
        case missingTensor(String)
        case unsupportedBits(Int)
        case badShape(String)
    }

    public let dir: URL
    public let defaultQuant: QuantSpec?
    /// Per-module overrides keyed by module path (e.g. "model.layers.0.mlp.gate").
    public let quantOverrides: [String: QuantSpec]

    private var files: [SafetensorsFile] = []
    private var fileURLs: [URL] = []
    /// The safetensors shards backing this checkpoint (model identity input).
    var shardURLs: [URL] { fileURLs }
    private var tensorToFile: [String: Int] = [:]

    public init(dir: URL) throws {
        self.dir = dir

        // Quantization config: {"quantization": {"group_size": 64, "bits": 4,
        // "model.layers.N.mlp.gate": {"group_size": 64, "bits": 8}, ...}}
        var defQuant: QuantSpec? = nil
        var overrides: [String: QuantSpec] = [:]
        let configURL = dir.appendingPathComponent("config.json")
        if let cfg = try? JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any],
           let q = cfg["quantization"] as? [String: Any] {
            if let g = q["group_size"] as? Int, let b = q["bits"] as? Int {
                defQuant = QuantSpec(groupSize: g, bits: b)
            }
            for (key, value) in q {
                if let sub = value as? [String: Any],
                   let g = sub["group_size"] as? Int, let b = sub["bits"] as? Int {
                    overrides[key] = QuantSpec(groupSize: g, bits: b)
                }
            }
        }
        defaultQuant = defQuant
        quantOverrides = overrides

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
        // Match the most specific override suffix (config keys omit shard prefixes).
        for (key, spec) in quantOverrides where path.hasSuffix(key) || key.hasSuffix(path) {
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
        let all = rawBytes(t)
        return all.withUnsafeBytes { buf in
            let words = buf.bindMemory(to: UInt32.self)
            if let r = elementRange { return Array(words[r]) }
            return Array(words)
        }
    }

    /// Float conversion over a sub-range of elements (F32/F16/BF16).
    func floats(_ name: String, elementRange: Range<Int>) throws -> [Float] {
        let t = try info(name)
        let bytes = rawBytes(t)
        switch t.dtype {
        case "F32":
            return bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)[elementRange]) }
        case "F16":
            return bytes.withUnsafeBytes { $0.bindMemory(to: Float16.self)[elementRange].map(Float.init) }
        case "BF16":
            return bytes.withUnsafeBytes {
                $0.bindMemory(to: UInt16.self)[elementRange].map { Float(bitPattern: UInt32($0) << 16) }
            }
        default:
            throw Error.unsupportedDtype(t.dtype, tensor: name)
        }
    }
}
