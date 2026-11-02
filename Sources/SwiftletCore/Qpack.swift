import Foundation

/// The `.qpack` container (v1), modeled on TurboFieldfare's `.gturbo`:
///
///   <name>.qpack/
///     manifest.json          arch + quant snapshot, file sizes
///     config.json            copied from the source checkpoint
///     model.safetensors      all dense/resident tensors, original quantized
///                            bytes (readable by `Checkpoint` as-is)
///     tokenizer files        copied from the source checkpoint
///     packed_experts/
///       layout.json          expert stride + intra-blob section offsets
///       layer_00.bin ...     one fixed-stride blob per expert, page aligned
///
/// One expert blob packs the six quantized sub-tensors of its three matrices
/// contiguously (gate/up/down x weight/scales/biases), so fetching an expert
/// is exactly one pread of `expertStride` bytes.
public enum Qpack {
    public static let pageAlignment = 16_384
    public static let manifestVersion = 1

    public struct Section: Codable, Sendable {
        public let name: String     // e.g. "gate_proj.weight"
        public let dtype: String
        public let shape: [Int]     // per-expert logical shape
        public let offset: Int      // byte offset inside the blob
        public let size: Int
    }

    public struct Layout: Codable, Sendable {
        public let expertCount: Int
        public let layerCount: Int
        public let expertStride: Int
        public let sections: [Section]
        public let linearLayers: [Bool]
    }

    public struct Manifest: Codable, Sendable {
        public let magic: String
        public let version: Int
        public let modelName: String
        public let sourceCheckpoint: String
        public let quantBits: Int?
        public let quantGroupSize: Int?
        public let files: [String: Int]   // relative path -> byte size
    }

    public enum Error: Swift.Error, CustomStringConvertible {
        /// No `manifest.json`: not a qpack container. The installer writes the
        /// manifest last, so this is also what an interrupted install looks like.
        case notAContainer(String)
        /// A file the manifest lists is absent.
        case missingFile(String)
        /// A file's size on disk disagrees with the manifest: a truncated or
        /// oversized copy, an interrupted transfer, or a file from another
        /// container.
        case sizeMismatch(path: String, expected: Int, actual: Int)
        /// `layout.json` promises `layerCount` blob files of
        /// `expertStride × expertCount` bytes; one disagrees.
        case layoutMismatch(String)
        /// The output directory already holds a checkpoint or a container;
        /// `file` is what says so. Neither producer writes over it.
        case occupiedOutput(dir: String, file: String)
        /// The output directory holds a streaming install that was
        /// interrupted (its progress sidecar is still there). Only the
        /// streaming installer can resume it; the repacker refuses.
        case interruptedInstall(String)

        public var description: String {
            switch self {
            case .notAContainer(let dir):
                return "\(dir) is not a qpack container (no manifest.json; an interrupted install looks the same way -- re-run it to resume)"
            case .occupiedOutput(let dir, let file):
                return "\(dir) already holds a checkpoint (\(file)); choose an empty output directory or remove it first"
            case .interruptedInstall(let dir):
                return "\(dir) holds an interrupted streaming install (\(Qpack.installProgressSidecar)); "
                    + "re-run the same install to resume it, or remove the directory before repacking into it"
            case .missingFile(let path):
                return "qpack container is missing \(path), which its manifest lists"
            case .sizeMismatch(let path, let expected, let actual):
                return "qpack container file \(path) is \(actual) bytes; its manifest recorded \(expected) "
                    + "(a truncated or foreign copy; re-download or re-repack the container)"
            case .layoutMismatch(let why):
                return "qpack container disagrees with its own packed_experts/layout.json: \(why)"
            }
        }
    }

    /// The streaming installer's progress sidecar. It is the first thing an
    /// install writes into its output directory and the last thing it
    /// removes, so its presence is what distinguishes an interrupted install
    /// of our own (resumable) from a checkpoint someone else put there.
    public static let installProgressSidecar = ".install-progress.json"

    /// Refuses `dir` as an output directory when it already holds a
    /// checkpoint or a container, naming the file that says so; an absent or
    /// empty directory, or one holding only aux files, is accepted. A
    /// finished container (`manifest.json`) or a sharded checkpoint's index
    /// is refused outright. A directory still carrying the install sidecar
    /// is an interrupted streaming install: `resumable` callers (the
    /// installer) proceed into it, others (the repacker) refuse it by name.
    /// Without the sidecar, any shard, expert layout or expert blob means
    /// the directory is someone's checkpoint, not an install of ours.
    public static func checkOutputDirectory(_ dir: URL, resumable: Bool) throws {
        let fm = FileManager.default
        // Absent (or unreadable): nothing to protect; the producer's own
        // createDirectory reports the latter.
        guard let names = try? fm.contentsOfDirectory(atPath: dir.path) else { return }
        let sorted = names.sorted()
        if sorted.contains("manifest.json") {
            throw Error.occupiedOutput(dir: dir.path, file: "manifest.json")
        }
        if let index = sorted.first(where: { $0.hasSuffix(".safetensors.index.json") }) {
            throw Error.occupiedOutput(dir: dir.path, file: index)
        }
        if sorted.contains(installProgressSidecar) {
            if resumable { return }
            throw Error.interruptedInstall(dir.path)
        }
        if let shard = sorted.first(where: { $0.hasSuffix(".safetensors") }) {
            throw Error.occupiedOutput(dir: dir.path, file: shard)
        }
        let expertsDir = dir.appendingPathComponent("packed_experts")
        if fm.fileExists(atPath: expertsDir.appendingPathComponent("layout.json").path) {
            throw Error.occupiedOutput(dir: dir.path, file: "packed_experts/layout.json")
        }
        if let blobs = try? fm.contentsOfDirectory(atPath: expertsDir.path),
           let blob = blobs.sorted().first(where: { $0.hasPrefix("layer_") && $0.hasSuffix(".bin") }) {
            throw Error.occupiedOutput(dir: dir.path, file: "packed_experts/" + blob)
        }
    }

    /// Verifies a container's structure from its own manifest and layout,
    /// before any expert byte is read: every file the manifest lists exists at
    /// the recorded size, and every `layer_NN.bin` is exactly
    /// `expertStride × expertCount` bytes. No receipt (`hashes.json`) is
    /// needed, so a container copied by hand is checked the same way as an
    /// installed one. Before this a short layer file opened cleanly and
    /// surfaced mid-generation as `short read: layer L expert E`, after the
    /// model had reported itself ready. Returns the verified layout.
    public static func verify(containerDir: URL) throws -> Layout {
        let fm = FileManager.default
        let manifestURL = containerDir.appendingPathComponent("manifest.json")
        guard fm.fileExists(atPath: manifestURL.path) else {
            throw Error.notAContainer(containerDir.path)
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        guard manifest.magic == "QPACK" else { throw Error.notAContainer(containerDir.path) }

        func sizeOnDisk(_ relative: String) throws -> Int {
            let path = containerDir.appendingPathComponent(relative).path
            guard fm.fileExists(atPath: path),
                  let size = try fm.attributesOfItem(atPath: path)[.size] as? Int else {
                throw Error.missingFile(relative)
            }
            return size
        }
        for (relative, expected) in manifest.files.sorted(by: { $0.key < $1.key }) {
            let actual = try sizeOnDisk(relative)
            guard actual == expected else {
                throw Error.sizeMismatch(path: relative, expected: expected, actual: actual)
            }
        }

        let layoutRelative = "packed_experts/layout.json"
        let layout = try JSONDecoder().decode(
            Layout.self, from: Data(contentsOf: containerDir.appendingPathComponent(layoutRelative)))
        let (perLayer, overflow) = layout.expertStride.multipliedReportingOverflow(by: layout.expertCount)
        guard !overflow, perLayer > 0, layout.layerCount > 0 else {
            throw Error.layoutMismatch(
                "\(layout.expertCount) experts × \(layout.expertStride)-byte stride × \(layout.layerCount) layers is not a container")
        }
        for layer in 0..<layout.layerCount {
            let relative = String(format: "packed_experts/layer_%02d.bin", layer)
            let actual = try sizeOnDisk(relative)
            guard actual == perLayer else {
                throw Error.layoutMismatch(
                    "\(relative) is \(actual) bytes; the layout promises \(layout.expertCount) experts × "
                    + "\(layout.expertStride)-byte stride = \(perLayer)")
            }
        }
        return layout
    }

    static func align(_ n: Int, to a: Int) -> Int { (n + a - 1) / a * a }

    /// Logical inner dimension of a quantized expert projection, cross-checked
    /// against the manifest's quantization. A quantized weight stores
    /// `inDim / (32 / bits)` uint32 words per row; its scales store
    /// `inDim / groupSize` groups per row. Both describe the same `inDim`, so
    /// when the manifest's bits/groupSize make them disagree the container's
    /// recorded quantization does not match its packed bytes, and dequantizing
    /// anyway would decode garbage. This catches an already-built bad container
    /// at load with a clear error instead of silent garbage (issue #30).
    static func expertLogicalInDim(weightLastDim: Int, scalesLastDim: Int,
                                   bits: Int, groupSize: Int,
                                   section: String) throws -> Int {
        guard bits > 0, groupSize > 0, 32 % bits == 0 else {
            throw Checkpoint.Error.badShape(
                "qpack section \(section): invalid quantization \(bits)-bit/g\(groupSize)")
        }
        let fromWeight = weightLastDim * (32 / bits)
        let fromScales = scalesLastDim * groupSize
        guard fromWeight > 0, fromWeight == fromScales else {
            throw Checkpoint.Error.badShape(
                "qpack section \(section): manifest \(bits)-bit/g\(groupSize) disagrees with the packed "
                + "shapes (weight inner \(weightLastDim), scales inner \(scalesLastDim)); the container's "
                + "recorded quantization does not match its expert bytes (issue #30)")
        }
        return fromWeight
    }
}

/// Repacks a local mlx-lm checkpoint directory into a `.qpack` container.
public struct QpackRepacker {
    public let checkpointDir: URL
    public let outputDir: URL
    public var log: (String) -> Void = { print($0) }

    public init(checkpointDir: URL, outputDir: URL) {
        self.checkpointDir = checkpointDir
        self.outputDir = outputDir
    }

    public enum Error: Swift.Error, CustomStringConvertible {
        /// The packed expert projections do not share one quantization, which a
        /// single-valued manifest cannot represent (issue #30).
        case mixedExpertQuant(String)
        public var description: String {
            switch self { case .mixedExpertQuant(let m): return m }
        }
    }

    static let expertTensorSuffixes = ["gate_proj", "up_proj", "down_proj"]

    /// The quantization the packed EXPERT tensors were stored with, which is
    /// what the runtime must dequantize them by, resolved from the checkpoint's
    /// per-tensor overrides and falling back to its default. This is not always
    /// the checkpoint's top-level default: a mixed-precision checkpoint (for
    /// example an 8-bit dense default with 4-bit `switch_mlp` experts, as in the
    /// DWQ variants) carries the expert precision as per-tensor overrides.
    /// Returns nil for an unquantized checkpoint. Throws when the expert
    /// projections disagree with each other, since the manifest records one
    /// value for all of them (issue #30).
    static func expertQuant(_ ckpt: Checkpoint) throws -> Checkpoint.QuantSpec? {
        try expertQuant(default: ckpt.defaultQuant,
                        contains: { ckpt.contains($0) },
                        spec: { ckpt.quantSpec(for: $0) })
    }

    /// The rule above over any source that can answer "is this tensor present"
    /// and "what spec does this module resolve to": an opened checkpoint, or
    /// the streaming installer's shard headers plus the config it fetched, so
    /// both producers record the same expert quantization for the same source.
    static func expertQuant(default defaultQuant: Checkpoint.QuantSpec?,
                            contains: (String) -> Bool,
                            spec: (String) -> Checkpoint.QuantSpec?) throws -> Checkpoint.QuantSpec? {
        var resolved: [(module: String, spec: Checkpoint.QuantSpec)] = []
        for proj in expertTensorSuffixes {
            // Layer 0 is representative: the section table is derived from it,
            // and mlx-lm quantizes a given module identically across layers.
            let module = "model.layers.0.mlp.switch_mlp." + proj
            guard contains(module + ".weight") else { continue }
            guard let spec = spec(module) else { continue }
            resolved.append((module, spec))
        }
        guard let first = resolved.first else { return defaultQuant }
        for entry in resolved
        where entry.spec.bits != first.spec.bits || entry.spec.groupSize != first.spec.groupSize {
            throw Error.mixedExpertQuant(
                "expert projection \(entry.module) is \(entry.spec.bits)-bit/g\(entry.spec.groupSize) "
                + "but \(first.module) is \(first.spec.bits)-bit/g\(first.spec.groupSize); a container "
                + "records one expert quantization and cannot represent both")
        }
        return first.spec
    }

    /// Follows a (possibly relative, possibly chained) symlink to the real file
    /// it points at, returning the input unchanged when it is not a symlink.
    /// Hugging Face snapshot dirs point aux files at ../../blobs/<hash> via a
    /// single relative link; resolving it lets the repacker copy real bytes
    /// instead of a link that dangles once the container moves (issue #11).
    static func resolveSymlink(_ url: URL, fileManager fm: FileManager = .default) -> URL {
        var current = url
        var hops = 0
        while hops < 16, let dest = try? fm.destinationOfSymbolicLink(atPath: current.path) {
            if (dest as NSString).isAbsolutePath {
                current = URL(fileURLWithPath: dest).standardizedFileURL
            } else {
                current = URL(fileURLWithPath: dest,
                              relativeTo: current.deletingLastPathComponent()).standardizedFileURL
            }
            hops += 1
        }
        return current
    }

    public func repack() throws {
        let fm = FileManager.default
        let config = try QwenConfig(url: checkpointDir.appendingPathComponent("config.json"))
        let ckpt = try Checkpoint(dir: checkpointDir)

        // Resolve (and validate) the expert quantization up front so a
        // mixed-precision checkpoint fails before packing gigabytes rather than
        // after, and so the manifest records the experts' real precision instead
        // of the checkpoint's dense default (issue #30).
        let expertQuant = try Self.expertQuant(ckpt)

        // The output directory is claimed only when it holds no checkpoint:
        // a raw checkpoint there would have its shard truncated to the dense
        // file, a finished container its manifest overwritten. The repacker
        // has no resume, so an interrupted streaming install is refused too.
        try Qpack.checkOutputDirectory(outputDir, resumable: false)
        let expertsDir = outputDir.appendingPathComponent("packed_experts")
        try fm.createDirectory(at: expertsDir, withIntermediateDirectories: true)

        // --- Expert layout: derive per-expert section table from layer 0. ---
        let l0 = "model.layers.0.mlp.switch_mlp."
        var sections: [Qpack.Section] = []
        var offset = 0
        for proj in Self.expertTensorSuffixes {
            for part in ["weight", "scales", "biases"] {
                let name = l0 + proj + "." + part
                guard ckpt.contains(name) else { continue }
                let (info, _) = try ckpt.rawTensor(name)
                guard info.shape.first == config.numExperts,
                      let perElem = SafetensorsFile.bytesPerElement(info.dtype)
                else { throw Checkpoint.Error.badShape(name) }
                let perExpertShape = Array(info.shape.dropFirst())
                let perExpertBytes = perExpertShape.reduce(1, *) * perElem
                sections.append(Qpack.Section(
                    name: proj + "." + part, dtype: info.dtype, shape: perExpertShape,
                    offset: offset, size: perExpertBytes
                ))
                offset += perExpertBytes
            }
        }
        // A checkpoint with no stacked expert tensors is not in mlx-lm runtime
        // format. Without this guard the loop above silently skips every section,
        // yielding zero-length layer files and a structurally valid container
        // that only fails much later, at load time.
        guard !sections.isEmpty else {
            throw Checkpoint.Error.missingTensor(
                l0 + "* -- no stacked expert tensors found. This is not an mlx-lm "
                + "runtime-format checkpoint; convert it with mlx-lm first "
                + "(raw Hugging Face checkpoints store experts as .mlp.experts.N.*)."
            )
        }

        let stride = Qpack.align(offset, to: Qpack.pageAlignment)
        let layout = Qpack.Layout(
            expertCount: config.numExperts,
            layerCount: config.numHiddenLayers,
            expertStride: stride,
            sections: sections,
            linearLayers: (0..<config.numHiddenLayers).map(config.isLinearLayer)
        )
        log("expert blob: \(offset) B payload, stride \(stride) B, \(sections.count) sections")

        // --- Pack layer files. ---
        var files: [String: Int] = [:]
        for layer in 0..<config.numHiddenLayers {
            let prefix = "model.layers.\(layer).mlp.switch_mlp."
            var blob = Data(count: stride * config.numExperts)
            for section in sections {
                let (info, bytes) = try ckpt.rawTensor(prefix + section.name)
                guard info.shape.first == config.numExperts else {
                    throw Checkpoint.Error.badShape(prefix + section.name)
                }
                let perExpert = section.size
                for e in 0..<config.numExperts {
                    let src = e * perExpert..<(e + 1) * perExpert
                    let dst = e * stride + section.offset
                    blob.replaceSubrange(dst..<dst + perExpert, with: bytes[src])
                }
            }
            let fileName = String(format: "layer_%02d.bin", layer)
            try blob.write(to: expertsDir.appendingPathComponent(fileName))
            files["packed_experts/" + fileName] = blob.count
            log("packed \(fileName) (\(blob.count / 1_048_576) MiB)")
        }

        let layoutData = try JSONEncoder.sorted.encode(layout)
        try layoutData.write(to: expertsDir.appendingPathComponent("layout.json"))
        files["packed_experts/layout.json"] = layoutData.count

        // --- Dense/resident file: everything except experts and MTP. ---
        var dense: [(name: String, dtype: String, shape: [Int], bytes: Data)] = []
        for name in ckpt.tensorNames.sorted() {
            if name.contains(".mlp.switch_mlp.") || name.hasPrefix("mtp.") || name.contains(".mtp.") { continue }
            if name.hasPrefix("vision_tower.") || name.contains("model.visual") { continue }
            let (info, bytes) = try ckpt.rawTensor(name)
            dense.append((name, info.dtype, info.shape, bytes))
        }
        let denseURL = outputDir.appendingPathComponent("model.safetensors")
        try SafetensorsFile.write(to: denseURL, tensors: dense)
        files["model.safetensors"] = try fm.attributesOfItem(atPath: denseURL.path)[.size] as? Int ?? 0
        log("dense file: \(dense.count) tensors, \(files["model.safetensors"]! / 1_048_576) MiB")

        // --- Copy config + tokenizer artifacts. ---
        for aux in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                    "merges.txt", "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"] {
            let src = checkpointDir.appendingPathComponent(aux)
            guard fm.fileExists(atPath: src.path) else { continue }
            let dst = outputDir.appendingPathComponent(aux)
            try? fm.removeItem(at: dst)
            // Hugging Face snapshot dirs store these as relative symlinks into
            // ../../blobs; copyItem preserves the link verbatim, so the
            // container ships a dangling symlink and swiftlet-server fails with
            // configurationMissing (issue #11). Resolve to the real file first
            // so the actual bytes land in the container.
            try fm.copyItem(at: Self.resolveSymlink(src, fileManager: fm), to: dst)
            files[aux] = try fm.attributesOfItem(atPath: dst.path)[.size] as? Int ?? 0
        }

        // quantBits/quantGroupSize describe the packed EXPERT tensors (resolved
        // and validated above), not the checkpoint's dense default (issue #30).
        let manifest = Qpack.Manifest(
            magic: "QPACK",
            version: Qpack.manifestVersion,
            modelName: config.modelType,
            sourceCheckpoint: checkpointDir.lastPathComponent,
            quantBits: expertQuant?.bits,
            quantGroupSize: expertQuant?.groupSize,
            files: files
        )
        try JSONEncoder.sorted.encode(manifest).write(to: outputDir.appendingPathComponent("manifest.json"))
        log("wrote manifest; container complete at \(outputDir.path)")
    }
}

/// Reads expert blobs back out of a `.qpack` container, one pread per expert.
public final class QpackExpertReader {
    public let layout: Qpack.Layout
    private let dir: URL
    private var fds: [Int32]

    public init(containerDir: URL) throws {
        dir = containerDir.appendingPathComponent("packed_experts")
        if FileManager.default.fileExists(atPath: containerDir.appendingPathComponent("manifest.json").path) {
            // A real container: check every listed file and every blob file
            // against the sizes the container itself recorded, before the
            // first pread. A directory carrying only packed_experts/ is a
            // test scaffold, opened as before; the model opener requires the
            // manifest regardless.
            layout = try Qpack.verify(containerDir: containerDir)
        } else {
            let layoutData = try Data(contentsOf: dir.appendingPathComponent("layout.json"))
            layout = try JSONDecoder().decode(Qpack.Layout.self, from: layoutData)
        }
        fds = Array(repeating: -1, count: layout.layerCount)
    }

    deinit { for fd in fds where fd >= 0 { close(fd) } }

    /// Single-pread fetch of one expert blob into a caller-provided buffer of
    /// at least `layout.expertStride` bytes.
    public func readExpert(layer: Int, expert: Int, into buffer: UnsafeMutableRawPointer) throws {
        let fd = try descriptor(layer: layer)
        try Self.readBlob(fd, layer: layer, expert: expert, stride: layout.expertStride, into: buffer)
    }

    /// Several expert blobs of one layer, each into its own buffer. Two or
    /// more are read concurrently (one pread per thread; distinct offsets,
    /// distinct destinations, one shared descriptor), which is what turns a
    /// decode step's clustered cache misses from serial SSD latency into
    /// queue depth. Any failure is rethrown after every read has stopped.
    public func readExperts(
        layer: Int, _ requests: [(expert: Int, into: UnsafeMutableRawPointer)]
    ) throws {
        guard !requests.isEmpty else { return }
        let fd = try descriptor(layer: layer)
        let stride = layout.expertStride
        if requests.count == 1 {
            try Self.readBlob(fd, layer: layer, expert: requests[0].expert, stride: stride,
                           into: requests[0].into)
            return
        }
        let lock = NSLock()
        var failure: Swift.Error?
        DispatchQueue.concurrentPerform(iterations: requests.count) { i in
            do {
                try Self.readBlob(fd, layer: layer, expert: requests[i].expert, stride: stride,
                               into: requests[i].into)
            } catch {
                lock.lock()
                if failure == nil { failure = error }
                lock.unlock()
            }
        }
        if let failure { throw failure }
    }

    private func descriptor(layer: Int) throws -> Int32 {
        if fds[layer] < 0 {
            let path = dir.appendingPathComponent(String(format: "layer_%02d.bin", layer)).path
            let fd = open(path, O_RDONLY)
            guard fd >= 0 else { throw Checkpoint.Error.missingTensor(path) }
            fds[layer] = fd
        }
        return fds[layer]
    }

    private static func readBlob(
        _ fd: Int32, layer: Int, expert: Int, stride: Int, into buffer: UnsafeMutableRawPointer
    ) throws {
        let n = pread(fd, buffer, stride, off_t(expert * stride))
        guard n == stride else {
            throw Checkpoint.Error.badShape("short read: layer \(layer) expert \(expert)")
        }
    }

    public func section(_ name: String) -> Qpack.Section? {
        layout.sections.first { $0.name == name }
    }
}

extension JSONEncoder {
    static var sorted: JSONEncoder {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys, .prettyPrinted]
        return e
    }
}
