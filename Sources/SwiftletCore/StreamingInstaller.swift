import Foundation
import CryptoKit

/// TurboFieldfare-style streaming installer: pulls a sharded MLX checkpoint
/// from Hugging Face (or a local directory) and routes every arriving byte
/// straight into its final position in the .qpack container — expert bytes
/// into the fixed-stride layer blobs, dense bytes into the resident weights
/// file. The raw checkpoint is never stored: peak disk = the container.
/// Resumable per shard via a progress sidecar.
public final class StreamingInstaller {

    // MARK: Byte sources

    public enum Source {
        case huggingFace(repo: String)
        /// Any static host serving the container files (e.g. an R2/CDN
        /// mirror) — same layout as the HF repo, no rate limiting.
        case baseURL(String)
        case localDirectory(URL)

        /// Remote URL prefix, nil for local sources.
        var httpBase: String? {
            switch self {
            case .huggingFace(let repo): return "https://huggingface.co/\(repo)/resolve/main"
            case .baseURL(let base): return base.hasSuffix("/") ? String(base.dropLast()) : base
            case .localDirectory: return nil
            }
        }

        func smallFile(_ name: String) throws -> Data? {
            switch self {
            case .localDirectory(let dir):
                return FileManager.default.contents(atPath: dir.appendingPathComponent(name).path)
            case .huggingFace, .baseURL:
                let url = URL(string: "\(httpBase!)/\(name)")!
                var req = URLRequest(url: url)
                req.timeoutInterval = 60
                StreamingInstaller.authorize(&req)
                let sem = DispatchSemaphore(value: 0)
                var result: Data?
                var status = 0
                URLSession.shared.dataTask(with: req) { d, r, _ in
                    status = (r as? HTTPURLResponse)?.statusCode ?? 0
                    result = d
                    sem.signal()
                }.resume()
                sem.wait()
                return status == 200 ? result : nil
            }
        }
    }

    /// Adds a Hugging Face bearer token from HF_TOKEN when present —
    /// authenticated requests get far higher rate limits.
    static func authorize(_ req: inout URLRequest) {
        if let token = ProcessInfo.processInfo.environment["HF_TOKEN"], !token.isEmpty {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
    }

    public enum Error: Swift.Error, LocalizedError {
        case missingFile(String)
        case badHeader(String)
        case httpError(String, Int)
        case ioError(String)
        case hashMismatch(String)
        case cancelled
        case notMLXCheckpoint

        public var errorDescription: String? {
            switch self {
            case .missingFile(let f): return "missing file \(f)"
            case .badHeader(let f): return "unreadable header in \(f)"
            case .httpError(let f, let c): return "network error \(c) on \(f)"
            case .ioError(let f): return "could not write \(f)"
            case .hashMismatch(let f): return "integrity check failed for \(f)"
            case .cancelled: return "cancelled"
            case .notMLXCheckpoint:
                return "no model.layers.0.mlp.switch_mlp.* tensors found -- this is not "
                    + "an mlx-lm runtime-format checkpoint. Convert it with mlx-lm first "
                    + "(raw Hugging Face checkpoints store experts as .mlp.experts.N.*)."
            }
        }
    }

    /// SHA-256 manifest published next to a packed container (`hashes.json`),
    /// the same file `scripts/verify_container.py --write-hashes` produces:
    /// { "files": { "<relative path>": "<hex sha256>", ... } }.
    private struct HashesFile: Decodable { let files: [String: String] }

    let source: Source
    let outputDir: URL
    public var log: (String) -> Void = { print($0) }
    /// Polled between chunks/files; return true to abort the install with
    /// Error.cancelled. Progress on disk is kept, so a later install resumes.
    public var shouldCancel: (() -> Bool)?
    /// Cumulative bytes written this run (direct-download path), reported
    /// about every 8 MB — drive UI progress from this, not from log lines.
    public var onBytes: ((Int) -> Void)?

    private let progressLock = NSLock()
    private var bytesThisRun = 0
    private var lastReportedBytes = 0

    private func reportBytes(_ n: Int) {
        progressLock.lock()
        bytesThisRun += n
        let total = bytesThisRun
        let due = total - lastReportedBytes >= 8 << 20
        if due { lastReportedBytes = total }
        progressLock.unlock()
        if due { onBytes?(total) }
    }

    public init(source: Source, outputDir: URL) {
        self.source = source
        self.outputDir = outputDir
    }

    // MARK: Plans

    struct TensorSpan {
        let name: String
        let start: Int      // absolute byte offset in shard file
        let end: Int
        enum Target {
            case dense(fileOffset: Int)                       // into dense file
            case expert(layer: Int, sectionOffset: Int, perExpertBytes: Int)
            case skip
        }
        var target: Target
    }

    struct ShardPlan {
        let name: String
        let size: Int
        var spans: [TensorSpan]   // sorted by start
    }

    // MARK: Install

    public func install() throws {
        let fm = FileManager.default
        // A directory that already holds a checkpoint is refused by name; the
        // only thing this installer writes into is an empty directory or an
        // interrupted install of its own, which the sidecar identifies.
        try Qpack.checkOutputDirectory(outputDir, resumable: true)
        try fm.createDirectory(at: outputDir.appendingPathComponent("packed_experts"), withIntermediateDirectories: true)
        // The sidecar goes down first and comes up last: from here until the
        // manifest is written the directory is an install in progress, and a
        // re-run resumes through it (per-shard byte cursors on the repacking
        // path, per-file sizes on the direct-download path). A re-run must
        // not reset the cursors already recorded.
        let progressURL = outputDir.appendingPathComponent(Qpack.installProgressSidecar)
        if !fm.fileExists(atPath: progressURL.path) {
            try Data("{}".utf8).write(to: progressURL)
        }

        // Already-packed container repo (our published .qpack repos): download
        // the files directly instead of repacking. Detected by manifest.json.
        if let mData = try source.smallFile("manifest.json"),
           let manifest = try? JSONDecoder().decode(Qpack.Manifest.self, from: mData),
           manifest.magic == "QPACK" {
            log("packed container repo detected — direct download")
            // SHA-256 of every file, published next to the container as
            // hashes.json. Present on our mirrors; if a source lacks it we
            // fall back to the manifest's byte-size check with a warning.
            let expectedHashes: [String: String] = {
                guard let d = try? source.smallFile("hashes.json"),
                      let h = try? JSONDecoder().decode(HashesFile.self, from: d)
                else { return [:] }
                return h.files
            }()
            if expectedHashes.isEmpty {
                log("no hashes.json at source, verifying by size only")
            } else {
                log("hashes.json found, verifying \(expectedHashes.count) files by SHA-256")
            }
            for aux in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                        "merges.txt", "chat_template.jinja", "generation_config.json",
                        "special_tokens_map.json", "added_tokens.json", "packed_experts/layout.json"] {
                if let d = try source.smallFile(aux) {
                    try d.write(to: outputDir.appendingPathComponent(aux))
                }
            }
            for (file, size) in manifest.files.sorted(by: { $0.key < $1.key })
            where file != "packed_experts/layout.json" {
                if shouldCancel?() == true { throw Error.cancelled }
                try downloadFile(file, expectedSize: size, expectedHash: expectedHashes[file])
            }
            // Manifest last: its presence marks the container complete.
            try mData.write(to: outputDir.appendingPathComponent("manifest.json"))
            try? fm.removeItem(at: progressURL)
            log("container complete at \(outputDir.path)")
            return
        }

        // 1. Small files.
        guard let configData = try source.smallFile("config.json") else { throw Error.missingFile("config.json") }
        // Refuse a non-affine checkpoint (mxfp4/nvfp4/mxfp8) now, before the
        // shard plan and the first weight byte: streaming it would fill a
        // container the runtime cannot dequantize. Same decision as
        // Checkpoint's, so the CLI reports the mode by name either way. The
        // parsed block is kept: the manifest's expert quantization is resolved
        // from it once the shard headers say which modules are quantized.
        let configObject = (try JSONSerialization.jsonObject(with: configData) as? [String: Any]) ?? [:]
        let quant = try Checkpoint.quantization(fromConfig: configObject)
        // Same for the family: a checkpoint of another model_type is refused
        // here, by name, before anything is written into the output directory.
        _ = try QwenConfig.modelType(fromConfig: configObject)
        try configData.write(to: outputDir.appendingPathComponent("config.json"))
        for aux in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                    "chat_template.jinja", "generation_config.json", "special_tokens_map.json", "added_tokens.json"] {
            if let d = try source.smallFile(aux) {
                try d.write(to: outputDir.appendingPathComponent(aux))
            }
        }
        let config = try QwenConfig(url: outputDir.appendingPathComponent("config.json"))

        // 2. Shard list + headers.
        var shardNames: [String] = []
        if let idx = try source.smallFile("model.safetensors.index.json"),
           let obj = try JSONSerialization.jsonObject(with: idx) as? [String: Any],
           let weightMap = obj["weight_map"] as? [String: String] {
            shardNames = Array(Set(weightMap.values)).sorted()
        } else {
            shardNames = ["model.safetensors"]
        }
        log("planning \(shardNames.count) shard(s)")

        struct RawTensor {
            let name: String
            let dtype: String
            let shape: [Int]
            let start: Int    // absolute in shard
            let end: Int
            let shard: Int
        }
        var all: [RawTensor] = []
        var shardSizes: [Int] = []
        for (si, shard) in shardNames.enumerated() {
            let head = try fetchHeader(shard: shard)
            shardSizes.append(head.fileSize)
            for (name, info) in head.tensors {
                all.append(RawTensor(
                    name: name, dtype: info.dtype, shape: info.shape,
                    start: head.dataStart + info.byteRange.lowerBound,
                    end: head.dataStart + info.byteRange.upperBound, shard: si
                ))
            }
        }

        func isExpert(_ n: String) -> Bool { n.contains(".mlp.switch_mlp.") }
        func isSkipped(_ n: String) -> Bool {
            n.hasPrefix("vision_tower.") || n.contains("model.visual") || n.hasPrefix("mtp.") || n.contains(".mtp.")
        }
        func canonical(_ n: String) -> String {
            n.hasPrefix("language_model.") ? String(n.dropFirst("language_model.".count)) : n
        }

        // 3a. Expert section table from layer 0 (same order as QpackRepacker).
        var sections: [Qpack.Section] = []
        var sectionOffsetByKey: [String: (offset: Int, perExpert: Int)] = [:]
        var running = 0
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            for part in ["weight", "scales", "biases"] {
                let key = proj + "." + part
                guard let t = all.first(where: {
                    canonical($0.name) == "model.layers.0.mlp.switch_mlp.\(key)"
                }) else { continue }
                guard t.shape.first == config.numExperts,
                      let perElem = SafetensorsFile.bytesPerElement(t.dtype)
                else { throw Error.badHeader(key) }
                let perShape = Array(t.shape.dropFirst())
                let perBytes = perShape.reduce(1, *) * perElem
                sections.append(Qpack.Section(name: key, dtype: t.dtype, shape: perShape, offset: running, size: perBytes))
                sectionOffsetByKey[key] = (running, perBytes)
                running += perBytes
            }
        }
        // Same reasoning as QpackRepacker: an empty section table means the source
        // is not an mlx-lm runtime-format checkpoint. Fail here rather than writing
        // a container that cannot be loaded.
        guard !sections.isEmpty else { throw Error.notMLXCheckpoint }

        // The expert quantization the manifest must record, resolved per module
        // exactly as QpackRepacker resolves it (issue #30): a mixed-precision
        // checkpoint (8-bit dense default, 4-bit `switch_mlp` experts, as the
        // DWQ variants) carries the expert precision as per-module overrides,
        // and recording the checkpoint default made the load-time cross-check
        // refuse the container this installer had just streamed. Decided here,
        // from the headers and the config, before any weight byte is
        // downloaded, so disagreeing expert projections refuse without
        // streaming anything.
        let canonicalNames = Set(all.map { canonical($0.name) })
        let expertQuant = try QpackRepacker.expertQuant(
            default: quant.default,
            contains: { canonicalNames.contains($0) },
            spec: { module in
                canonicalNames.contains(module + ".scales")
                    ? Checkpoint.quantSpec(for: module, default: quant.default, overrides: quant.overrides)
                    : nil
            })

        let stride = Qpack.align(running, to: Qpack.pageAlignment)
        log("expert blob payload \(running) B, stride \(stride) B")

        // 3b. Dense plan: sorted canonical names -> contiguous final layout.
        let denseTensors = all.filter { !isExpert($0.name) && !isSkipped($0.name) }
            .sorted { canonical($0.name) < canonical($1.name) }
        var denseHeader: [String: Any] = [:]
        var denseOffset = 0
        var denseOffsets: [String: Int] = [:]
        for t in denseTensors {
            let size = t.end - t.start
            denseHeader[canonical(t.name)] = [
                "dtype": t.dtype, "shape": t.shape,
                "data_offsets": [denseOffset, denseOffset + size],
            ]
            denseOffsets[t.name] = denseOffset
            denseOffset += size
        }
        let denseHeaderData = try JSONSerialization.data(withJSONObject: denseHeader)
        let denseDataStart = 8 + denseHeaderData.count

        // 4. Open output files.
        let denseURL = outputDir.appendingPathComponent("model.safetensors")
        let denseFd = open(denseURL.path, O_RDWR | O_CREAT, 0o644)
        guard denseFd >= 0 else { throw Error.ioError("dense file") }
        defer { close(denseFd) }
        ftruncate(denseFd, off_t(denseDataStart + denseOffset))
        var lenLE = UInt64(denseHeaderData.count).littleEndian
        withUnsafeBytes(of: &lenLE) { _ = pwrite(denseFd, $0.baseAddress!, 8, 0) }
        _ = denseHeaderData.withUnsafeBytes { pwrite(denseFd, $0.baseAddress!, denseHeaderData.count, 8) }

        var layerFds: [Int32] = []
        for l in 0..<config.numHiddenLayers {
            let path = outputDir.appendingPathComponent(String(format: "packed_experts/layer_%02d.bin", l)).path
            let fd = open(path, O_RDWR | O_CREAT, 0o644)
            guard fd >= 0 else { throw Error.ioError("layer \(l)") }
            ftruncate(fd, off_t(stride * config.numExperts))
            layerFds.append(fd)
        }
        defer { for fd in layerFds where fd >= 0 { close(fd) } }

        // 5. Build per-shard span plans.
        var plans: [ShardPlan] = []
        for (si, shard) in shardNames.enumerated() {
            var spans: [TensorSpan] = []
            for t in all where t.shard == si {
                var target: TensorSpan.Target = .skip
                if isSkipped(t.name) {
                    target = .skip
                } else if isExpert(t.name) {
                    let c = canonical(t.name)
                    // model.layers.N.mlp.switch_mlp.<proj>.<part>
                    let parts = c.split(separator: ".")
                    guard let layerIdx = parts.firstIndex(of: "layers").map({ Int(parts[$0 + 1])! }),
                          parts.count >= 2 else { throw Error.badHeader(t.name) }
                    let key = parts.suffix(2).joined(separator: ".")
                    guard let sec = sectionOffsetByKey[key] else { throw Error.badHeader(key) }
                    target = .expert(layer: layerIdx, sectionOffset: sec.offset, perExpertBytes: sec.perExpert)
                } else {
                    target = .dense(fileOffset: denseDataStart + denseOffsets[t.name]!)
                }
                spans.append(TensorSpan(name: t.name, start: t.start, end: t.end, target: target))
            }
            spans.sort { $0.start < $1.start }
            plans.append(ShardPlan(name: shard, size: shardSizes[si], spans: spans))
        }

        // 6. Stream shards, routing bytes; resume via sidecar.
        var progress: [String: Int] = (try? JSONSerialization.jsonObject(
            with: Data(contentsOf: progressURL))) as? [String: Int] ?? [:]

        for plan in plans {
            let done = progress[plan.name] ?? 0
            if done >= plan.size {
                log("\(plan.name): already complete")
                continue
            }
            log("\(plan.name): streaming from byte \(done) of \(plan.size)")
            var cursor = done
            try streamShard(plan.name, from: done) { chunk in
                try self.route(chunk: chunk, at: &cursor, plan: plan,
                               denseFd: denseFd, layerFds: layerFds, stride: stride)
                progress[plan.name] = cursor
                if cursor % (256 * 1024 * 1024) < chunk.count {
                    try? JSONSerialization.data(withJSONObject: progress).write(to: progressURL)
                    self.log(String(format: "  %@ %.0f%%", plan.name, 100 * Double(cursor) / Double(plan.size)))
                }
            }
            guard cursor == plan.size else { throw Error.ioError("\(plan.name) short stream at \(cursor)") }
            progress[plan.name] = cursor
            try JSONSerialization.data(withJSONObject: progress).write(to: progressURL)
        }

        // 7. Layout + manifest.
        let layout = Qpack.Layout(
            expertCount: config.numExperts,
            layerCount: config.numHiddenLayers,
            expertStride: stride,
            sections: sections,
            linearLayers: (0..<config.numHiddenLayers).map(config.isLinearLayer)
        )
        try JSONEncoder.sorted.encode(layout)
            .write(to: outputDir.appendingPathComponent("packed_experts/layout.json"))
        var files: [String: Int] = ["model.safetensors": denseDataStart + denseOffset]
        for l in 0..<config.numHiddenLayers {
            files[String(format: "packed_experts/layer_%02d.bin", l)] = stride * config.numExperts
        }
        let manifest = Qpack.Manifest(
            magic: "QPACK", version: Qpack.manifestVersion,
            modelName: config.modelType,
            sourceCheckpoint: {
                if case .huggingFace(let repo) = source { return repo }
                if case .baseURL(let base) = source { return base }
                return "local"
            }(),
            quantBits: expertQuant?.bits,
            quantGroupSize: expertQuant?.groupSize,
            files: files
        )
        try JSONEncoder.sorted.encode(manifest).write(to: outputDir.appendingPathComponent("manifest.json"))
        try? fm.removeItem(at: progressURL)
        log("container complete at \(outputDir.path)")
    }

    // MARK: Routing

    private func route(chunk: Data, at cursor: inout Int, plan: ShardPlan,
                       denseFd: Int32, layerFds: [Int32], stride: Int) throws {
        var offset = 0
        let count = chunk.count
        try chunk.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let base = raw.baseAddress!
            while offset < count {
                let absolute = cursor + offset
                // Find the span containing `absolute` (spans sorted; linear scan
                // with memo would be nicer, binary search is fine at this rate).
                var lo = 0, hi = plan.spans.count
                while lo < hi {
                    let mid = (lo + hi) / 2
                    if plan.spans[mid].end <= absolute { lo = mid + 1 } else { hi = mid }
                }
                guard lo < plan.spans.count else {
                    offset = count   // trailing bytes past last tensor (padding)
                    break
                }
                let span = plan.spans[lo]
                if absolute < span.start {
                    // Gap (header/padding bytes between tensors): discard.
                    offset += min(span.start - absolute, count - offset)
                    continue
                }
                let rel = absolute - span.start
                let n = min(span.end - absolute, count - offset)
                switch span.target {
                case .skip:
                    offset += n
                case .dense(let fileOffset):
                    let wrote = pwrite(denseFd, base + offset, n, off_t(fileOffset + rel))
                    guard wrote == n else { throw Error.ioError("dense pwrite") }
                    offset += n
                case .expert(let layer, let sectionOffset, let perExpert):
                    var remaining = n
                    var r = rel
                    var chunkOff = offset
                    while remaining > 0 {
                        let e = r / perExpert
                        let within = r % perExpert
                        let take = min(perExpert - within, remaining)
                        let dst = e * stride + sectionOffset + within
                        let wrote = pwrite(layerFds[layer], base + chunkOff, take, off_t(dst))
                        guard wrote == take else { throw Error.ioError("expert pwrite") }
                        r += take
                        chunkOff += take
                        remaining -= take
                    }
                    offset += n
                }
            }
        }
        cursor += count
    }

    /// Direct download of one container file with byte-offset resume. When
    /// `expectedHash` is given, the finished file is checked against it and a
    /// mismatch deletes the file and throws, so a later run re-fetches it.
    private func downloadFile(_ name: String, expectedSize: Int, expectedHash: String?) throws {
        let dest = outputDir.appendingPathComponent(name)
        try FileManager.default.createDirectory(
            at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        let existing = (try? FileManager.default.attributesOfItem(atPath: dest.path)[.size] as? Int) ?? 0
        if existing == expectedSize {
            // Right size already on disk. Trust it only if the hash checks out
            // (or there is nothing to check against); otherwise re-download.
            if let expectedHash {
                if (try? sha256Hex(ofFileAt: dest)) == expectedHash {
                    log("\(name): already complete (verified)")
                    return
                }
                log("\(name): cached copy failed integrity check, re-downloading")
            } else {
                log("\(name): already complete")
                return
            }
        }
        let start = existing
        // A truncated-garbage or corrupt file restarts from zero.
        let resumeFrom = (existing > 0 && existing < expectedSize) ? existing : 0
        let fd = open(dest.path, O_WRONLY | O_CREAT, 0o644)
        guard fd >= 0 else { throw Error.ioError(name) }
        defer { close(fd) }
        if resumeFrom == 0 { ftruncate(fd, 0) }
        var cursor = resumeFrom
        // Incremental SHA-256 is only valid from byte 0: a resume cannot see the
        // bytes already on disk, so in that case we re-read the file at the end.
        var hasher: SHA256? = (expectedHash != nil && resumeFrom == 0) ? SHA256() : nil
        _ = start
        log("\(name): downloading from byte \(cursor) of \(expectedSize)")
        try streamShard(name, from: cursor) { chunk in
            let wrote = chunk.withUnsafeBytes { pwrite(fd, $0.baseAddress!, chunk.count, off_t(cursor)) }
            guard wrote == chunk.count else { throw Error.ioError(name) }
            hasher?.update(data: chunk)
            cursor += chunk.count
            self.reportBytes(chunk.count)
            if cursor % (256 * 1024 * 1024) < chunk.count {
                self.log(String(format: "  %@ %.0f%%", name, 100 * Double(cursor) / Double(expectedSize)))
            }
        }
        guard cursor == expectedSize else { throw Error.ioError("\(name) short at \(cursor)") }

        if let expectedHash {
            let got: String
            if let hasher {
                got = hasher.finalize().map { String(format: "%02x", $0) }.joined()
            } else {
                got = try sha256Hex(ofFileAt: dest)   // resumed: hash the whole file
            }
            if got != expectedHash {
                try? FileManager.default.removeItem(at: dest)
                throw Error.hashMismatch(name)
            }
            log("\(name): verified")
        }
    }

    /// Streaming SHA-256 of a file on disk, returned as lowercase hex.
    private func sha256Hex(ofFileAt url: URL) throws -> String {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw Error.ioError(url.lastPathComponent) }
        defer { close(fd) }
        var hasher = SHA256()
        let bufSize = 4 << 20
        var buf = [UInt8](repeating: 0, count: bufSize)
        while true {
            let n = buf.withUnsafeMutableBytes { read(fd, $0.baseAddress, bufSize) }
            if n < 0 { throw Error.ioError(url.lastPathComponent) }
            if n == 0 { break }
            buf.withUnsafeBytes { raw in
                hasher.update(bufferPointer: UnsafeRawBufferPointer(rebasing: raw[0..<n]))
            }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    // MARK: Shard byte streams

    private struct ShardHeader {
        let fileSize: Int
        let dataStart: Int
        let tensors: [String: SafetensorsFile.TensorInfo]
    }

    private func fetchHeader(shard: String) throws -> ShardHeader {
        switch source {
        case .localDirectory(let dir):
            let url = dir.appendingPathComponent(shard)
            let f = try SafetensorsFile(url: url)
            let size = (try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
            let ds = try f.absoluteOffset(f.tensors.keys.first!) - f.tensors[f.tensors.keys.first!]!.byteRange.lowerBound
            return ShardHeader(fileSize: size, dataStart: ds, tensors: f.tensors)
        case .huggingFace, .baseURL:
            guard let first = try ranged(shard, 0..<8) else { throw Error.missingFile(shard) }
            let headerLen = first.withUnsafeBytes { Int($0.loadUnaligned(fromByteOffset: 0, as: UInt64.self)) }
            guard let headerData = try ranged(shard, 8..<(8 + headerLen)) else { throw Error.badHeader(shard) }
            guard let obj = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
                throw Error.badHeader(shard)
            }
            var tensors: [String: SafetensorsFile.TensorInfo] = [:]
            var maxEnd = 0
            for (name, v) in obj where name != "__metadata__" {
                guard let e = v as? [String: Any],
                      let dtype = e["dtype"] as? String,
                      let shape = e["shape"] as? [Int],
                      let offs = e["data_offsets"] as? [Int], offs.count == 2
                else { throw Error.badHeader(shard) }
                tensors[name] = SafetensorsFile.TensorInfo(dtype: dtype, shape: shape, byteRange: offs[0]..<offs[1])
                maxEnd = max(maxEnd, offs[1])
            }
            return ShardHeader(fileSize: 8 + headerLen + maxEnd, dataStart: 8 + headerLen, tensors: tensors)
        }
    }

    private func ranged(_ file: String, _ range: Range<Int>) throws -> Data? {
        guard let base = source.httpBase else { return nil }
        let url = URL(string: "\(base)/\(file)")!
        var req = URLRequest(url: url)
        req.setValue("bytes=\(range.lowerBound)-\(range.upperBound - 1)", forHTTPHeaderField: "Range")
        req.timeoutInterval = 60
        StreamingInstaller.authorize(&req)
        let sem = DispatchSemaphore(value: 0)
        var result: Data?
        URLSession.shared.dataTask(with: req) { d, r, _ in
            let code = (r as? HTTPURLResponse)?.statusCode ?? 0
            result = (code == 206 || code == 200) ? d : nil
            sem.signal()
        }.resume()
        sem.wait()
        return result
    }

    /// Streams shard bytes from `offset`, delivering chunks in order. Retries
    /// transient failures, resuming from the caller-advanced cursor.
    private func streamShard(_ shard: String, from offset: Int, handler: @escaping (Data) throws -> Void) throws {
        let guarded: (Data) throws -> Void = { [weak self] chunk in
            if self?.shouldCancel?() == true { throw Error.cancelled }
            try handler(chunk)
        }
        switch source {
        case .localDirectory(let dir):
            guard let fh = FileHandle(forReadingAtPath: dir.appendingPathComponent(shard).path) else {
                throw Error.missingFile(shard)
            }
            defer { try? fh.close() }
            try fh.seek(toOffset: UInt64(offset))
            while let d = try fh.read(upToCount: 4 * 1024 * 1024), !d.isEmpty {
                try guarded(d)
            }
        case .huggingFace, .baseURL:
            var attempt = 0
            var start = offset
            var lastError: Swift.Error?
            while attempt < 8 {
                let pump = HTTPChunkPump(
                    url: URL(string: "\(source.httpBase!)/\(shard)")!,
                    fromByte: start
                )
                do {
                    var delivered = 0
                    try pump.run { chunk in
                        try guarded(chunk)
                        delivered += chunk.count
                    }
                    return   // completed
                } catch {
                    if case Error.cancelled = error { throw error }   // user asked; don't retry
                    lastError = error
                    start += pump.bytesDelivered
                    attempt += 1
                    log("  network hiccup on \(shard) (attempt \(attempt)); resuming at \(start)")
                    Thread.sleep(forTimeInterval: min(30, Double(attempt) * 5))
                }
            }
            throw lastError ?? Error.httpError(shard, 0)
        }
    }
}

/// Bridges URLSession's delegate callbacks into a blocking, in-order chunk
/// stream with bounded buffering.
final class HTTPChunkPump: NSObject, URLSessionDataDelegate {
    private let url: URL
    private let fromByte: Int
    private var queue: [Data] = []
    private var finished = false
    private var failure: Error?
    private let lock = NSCondition()
    private(set) var bytesDelivered = 0

    init(url: URL, fromByte: Int) {
        self.url = url
        self.fromByte = fromByte
    }

    func run(_ handler: (Data) throws -> Void) throws {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 120
        let session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        var req = URLRequest(url: url)
        if fromByte > 0 {
            req.setValue("bytes=\(fromByte)-", forHTTPHeaderField: "Range")
        }
        StreamingInstaller.authorize(&req)
        session.dataTask(with: req).resume()

        // Idle timeout: a connection that dies without a delegate callback
        // would otherwise block this wait forever (observed: stream stalls at
        // 0 B/s with the process alive). Timing out turns the stall into an
        // error, and the caller's retry loop reconnects with a Range resume.
        // 90s: HF's CDN visibly decays long-lived streams to ~0 B/s within
        // minutes; a fresh connection restores full speed, so cycle fast.
        let stallLimit: TimeInterval = 90
        while true {
            lock.lock()
            var timedOut = false
            let deadline = Date(timeIntervalSinceNow: stallLimit)
            while queue.isEmpty && !finished && failure == nil {
                if !lock.wait(until: deadline) { timedOut = true; break }
            }
            let chunk = queue.isEmpty ? nil : queue.removeFirst()
            let done = finished && queue.isEmpty && chunk == nil
            let err = failure
            lock.unlock()
            if let err { throw err }
            if chunk == nil, !done, timedOut {
                throw StreamingInstaller.Error.httpError(
                    "stalled connection (no data for \(Int(stallLimit))s)", 0)
            }
            if let chunk {
                try handler(chunk)
                bytesDelivered += chunk.count
            }
            if done { return }
        }
    }

    // Validate the response BEFORE any body bytes are accepted. Without this,
    // a rate-limit or CDN error page (429/503 HTML) gets written straight
    // into the container as if it were weights — silent corruption that only
    // shows up as degraded model output. On resume (fromByte > 0) the server
    // must honor the Range with a 206: a 200 would restart the file from
    // byte zero while we append at an offset.
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        let ok = fromByte > 0 ? code == 206 : (code == 200 || code == 206)
        if ok {
            completionHandler(.allow)
            return
        }
        lock.lock()
        failure = StreamingInstaller.Error.httpError(url.lastPathComponent, code)
        finished = true
        lock.signal()
        lock.unlock()
        completionHandler(.cancel)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        lock.lock()
        queue.append(data)
        lock.signal()
        lock.unlock()
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        lock.lock()
        if let error { failure = error }
        finished = true
        lock.signal()
        lock.unlock()
    }
}
