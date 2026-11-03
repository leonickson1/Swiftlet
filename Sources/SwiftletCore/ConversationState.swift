import CryptoKit
import Foundation

/// Why a snapshot was refused. Every case is a hard stop: nothing is
/// restored from bytes that fail any check, and nothing is written from a
/// live state that could not be restored.
public enum ConversationStateError: Swift.Error, Equatable {
    /// Not a Swiftlet state file (or shorter than its magic).
    case badMagic
    /// A format version this build does not read.
    case unsupportedVersion(found: UInt32, supported: UInt32)
    /// Fewer bytes than the header says the file holds.
    case truncated
    /// Structurally wrong bytes: digest mismatch, bad section table, extra
    /// or missing layers, a section whose size does not fit the geometry.
    case corrupt(String)
    /// Written from another checkpoint (identity digests differ, hex).
    case modelMismatch(expected: String, found: String)
    /// Same checkpoint family but a state-shaping dimension differs.
    case geometryMismatch(field: String, expected: Int, found: Int)
    /// The RoPE base differs (compared bitwise): the cached K rows were
    /// rotated under another frequency table.
    case ropeThetaMismatch(expected: Double, found: Double)
    /// The elements are not float32.
    case dtypeMismatch(found: UInt32)
    /// The snapshot's position does not fit the restoring engine's
    /// `contextCapacity`. A position equal to the capacity restores; the
    /// next step is then refused by admission, as it is today.
    case capacityExceeded(position: Int, capacity: Int)
    /// The live state cannot be written: a step threw after it had begun
    /// mutating the state, another engine instance holds it bound (its
    /// recurrence is in that instance's GPU buffers), or a field does not
    /// fit the state's position.
    case inconsistentState(String)
}

/// The versioned binary snapshot of a `QwenCPUModel.DecodeState`. See
/// docs/STATE_FORMAT.md for the byte layout; the constants and offsets there
/// are pinned by ConversationStateTests.headerLayoutIsPinned.
///
/// Container: fixed 144-byte little-endian header, a run of float32
/// sections, and a trailing SHA-256 over everything before it. The header
/// carries the format version, dtype, a model identity digest (with the
/// scheme that produced it), the eleven geometry fields that shape the
/// state, the position, the section count, the total byte length, the
/// rotary split and RoPE base the K rows were rotated under, and the
/// writing engine's context capacity, so a reader can refuse a truncated,
/// foreign, or mis-shaped file before it interprets a single section.
public enum ConversationState {
    public static let formatVersion: UInt32 = 2
    static let magic: [UInt8] = Array("SWLSTATE".utf8)
    static let headerBytes = 144
    static let trailerBytes = 32
    /// Element dtype codes. Only float32 exists in v2.
    static let dtypeFloat32: UInt32 = 1

    /// What a section holds. Layouts match both engines' state:
    /// K/V rows [pos][kvHead][headDim], conv tail [convKernel-1][convDim],
    /// delta state [vHead][vDim][kDim].
    enum Kind: UInt32 {
        case k = 1
        case v = 2
        case convTail = 3
        case deltaState = 4
    }

    struct Section: Equatable {
        let layer: Int
        let kind: Kind
        let values: [Float]
    }

    /// The state-shaping dimensions, in header order. Anything else in the
    /// config (experts, MoE sizes) does not change what a state holds and
    /// is covered by the identity digest instead; the rotary split and RoPE
    /// base travel as their own header fields.
    struct Geometry: Equatable {
        static let fieldNames = [
            "numHiddenLayers", "fullAttentionInterval", "numKeyValueHeads", "headDim",
            "linearNumValueHeads", "linearNumKeyHeads", "linearKeyHeadDim",
            "linearValueHeadDim", "linearConvKernelDim", "hiddenSize", "vocabSize",
        ]
        /// Format rule (v2): each of these eleven fields, and `sectionCount`,
        /// is between 1 and this, inclusive. The reader enforces it before
        /// any arithmetic on the fields, and every section size derived from
        /// them below is overflow-checked as well, so a crafted file (whose
        /// trailer digest is free to compute) is refused as `corrupt` instead
        /// of trapping the process. Real geometries are orders of magnitude
        /// smaller (see docs/STATE_FORMAT.md).
        static let fieldCeiling = 1 << 24
        let values: [Int]

        init(values: [Int]) {
            precondition(values.count == Self.fieldNames.count)
            self.values = values
        }

        init(config c: QwenConfig) {
            self.init(values: [
                c.numHiddenLayers, c.fullAttentionInterval, c.numKeyValueHeads, c.headDim,
                c.linearNumValueHeads, c.linearNumKeyHeads, c.linearKeyHeadDim,
                c.linearValueHeadDim, c.linearConvKernelDim, c.hiddenSize, c.vocabSize,
            ])
        }

        var numHiddenLayers: Int { values[0] }
        var fullAttentionInterval: Int { values[1] }
        func kvRowFloats() throws -> Int { try Self.product(values[2], values[3]) }
        func convTailFloats() throws -> Int { try Self.product(values[8] - 1, convDim()) }
        func deltaStateFloats() throws -> Int { try Self.product(Self.product(values[4], values[7]), values[6]) }
        private func convDim() throws -> Int {
            try Self.sum(Self.product(Self.product(2, values[5]), values[6]), Self.product(values[4], values[7]))
        }
        func isLinearLayer(_ index: Int) -> Bool { (index + 1) % fullAttentionInterval != 0 }

        /// Overflow-checked arithmetic on header fields: a product or sum
        /// that does not fit is a `ShapeProblem` ("geometry overflow"), which
        /// the reader reports as `corrupt`, never a trap.
        static func product(_ a: Int, _ b: Int) throws -> Int {
            let (value, overflow) = a.multipliedReportingOverflow(by: b)
            guard !overflow else { throw ShapeProblem(message: "geometry overflow") }
            return value
        }

        static func sum(_ a: Int, _ b: Int) throws -> Int {
            let (value, overflow) = a.addingReportingOverflow(b)
            guard !overflow else { throw ShapeProblem(message: "geometry overflow") }
            return value
        }
    }

    struct SectionKey: Hashable {
        let layer: Int
        let kind: Kind
    }

    /// A decoded file: everything the header and sections say, before it
    /// is compared with any model.
    struct Snapshot {
        var identityScheme: UInt32
        var identity: [UInt8]
        var geometry: Geometry
        var rotaryDims: Int
        var writerContextCapacity: Int
        var ropeThetaBits: UInt64
        var position: Int
        var sections: [Section]

        func section(layer: Int, kind: Kind) -> Section? {
            sections.first { $0.layer == layer && $0.kind == kind }
        }
    }

    /// What a file says about itself, readable without a model (tooling,
    /// the CLI's "loaded state" line). Runs the full structural decode, so
    /// a corrupt or truncated file is refused here as well.
    public struct Info: Equatable, Sendable {
        public let identityScheme: UInt32
        public let identityHex: String
        public let position: Int
        /// `contextCapacity` of the engine that wrote the file. Informational:
        /// an engine with a larger capacity may restore the state.
        public let writerContextCapacity: Int
        public let byteCount: Int
    }

    public static func info(of data: Data) throws -> Info {
        let s = try decode(data)
        return Info(identityScheme: s.identityScheme,
                    identityHex: s.identity.map { String(format: "%02x", $0) }.joined(),
                    position: s.position, writerContextCapacity: s.writerContextCapacity,
                    byteCount: data.count)
    }

    // MARK: - Section shapes (shared by the writer and the reader)

    /// A section that cannot exist, a layer whose sections are missing, or a
    /// header whose derived sizes overflow. The reader reports it as
    /// `corrupt`, the writer as `inconsistentState`.
    struct ShapeProblem: Swift.Error {
        let message: String
    }

    /// The float count a (layer, kind) section must hold under `geometry`
    /// at `position`. Overflow-checked: the inputs are header bytes.
    static func expectedCount(layer: Int, kind: Kind, geometry: Geometry, position: Int) throws -> Int {
        guard layer < geometry.numHiddenLayers else {
            throw ShapeProblem(message: "section layer \(layer)")
        }
        switch kind {
        case .k, .v:
            guard !geometry.isLinearLayer(layer) else {
                throw ShapeProblem(message: "KV section on DeltaNet layer \(layer)")
            }
            return try Geometry.product(position, geometry.kvRowFloats())
        case .convTail:
            guard geometry.isLinearLayer(layer) else {
                throw ShapeProblem(message: "conv tail on attention layer \(layer)")
            }
            return try geometry.convTailFloats()
        case .deltaState:
            guard geometry.isLinearLayer(layer) else {
                throw ShapeProblem(message: "delta state on attention layer \(layer)")
            }
            return try geometry.deltaStateFloats()
        }
    }

    /// Every layer must be fully present once the sequence has started.
    static func missingSection(in sections: [Section], geometry: Geometry, position: Int) -> String? {
        guard position > 0 else { return nil }
        let present = Set(sections.map { SectionKey(layer: $0.layer, kind: $0.kind) })
        for li in 0..<geometry.numHiddenLayers {
            let kinds: [Kind] = geometry.isLinearLayer(li) ? [.convTail, .deltaState] : [.k, .v]
            for kind in kinds where !present.contains(SectionKey(layer: li, kind: kind)) {
                return "layer \(li) missing \(kind)"
            }
        }
        return nil
    }

    // MARK: - Writing

    /// Serializes `state` for `engine`, refusing anything a reader would
    /// not restore: a state whose last step threw after it had begun
    /// mutating it, a state another engine instance holds bound (its live
    /// recurrence is in that instance's GPU buffers; the fields here are a
    /// stale copy), or fields that do not fit the position. Callers on the
    /// Metal fast path capture the bound state's recurrence first.
    static func snapshot(
        of state: QwenCPUModel.DecodeState, engine: AnyObject, identity: ModelIdentity,
        config: QwenConfig, writerContextCapacity: Int
    ) throws -> Data {
        guard !state.tainted else {
            throw ConversationStateError.inconsistentState("step threw mid-token")
        }
        if let holder = state.boundOn, holder !== engine {
            throw ConversationStateError.inconsistentState("state is bound on another engine instance")
        }
        let geometry = Geometry(config: config)
        let sections = state.snapshotSections()
        for sec in sections {
            let expected: Int
            do {
                expected = try expectedCount(
                    layer: sec.layer, kind: sec.kind, geometry: geometry, position: state.position)
            } catch let problem as ShapeProblem {
                throw ConversationStateError.inconsistentState(problem.message)
            }
            guard sec.values.count == expected else {
                throw ConversationStateError.inconsistentState(
                    "layer \(sec.layer) \(sec.kind) holds \(sec.values.count) floats, geometry says \(expected)")
            }
        }
        if let missing = missingSection(in: sections, geometry: geometry, position: state.position) {
            throw ConversationStateError.inconsistentState(missing)
        }
        return encode(Snapshot(
            identityScheme: identity.scheme, identity: identity.digest, geometry: geometry,
            rotaryDims: config.rotaryDims, writerContextCapacity: writerContextCapacity,
            ropeThetaBits: config.ropeTheta.bitPattern, position: state.position, sections: sections))
    }

    static func encode(_ s: Snapshot) -> Data {
        precondition(s.identity.count == 32)
        var sectionBytes = 0
        for sec in s.sections { sectionBytes += 16 + 4 * sec.values.count }
        let total = headerBytes + sectionBytes + trailerBytes
        var body = Data(capacity: total - trailerBytes)
        body.append(contentsOf: magic)
        body.appendLE(formatVersion)
        body.appendLE(UInt32(headerBytes))
        body.appendLE(dtypeFloat32)
        body.appendLE(s.identityScheme)
        body.append(contentsOf: s.identity)
        for v in s.geometry.values { body.appendLE(UInt32(v)) }
        body.appendLE(UInt64(s.position))
        body.appendLE(UInt32(s.sections.count))
        body.appendLE(UInt64(total))
        body.appendLE(UInt32(s.rotaryDims))
        body.appendLE(UInt32(s.writerContextCapacity))
        body.appendLE(s.ropeThetaBits)
        body.append(contentsOf: [UInt8](repeating: 0, count: headerBytes - body.count))
        precondition(body.count == headerBytes)
        for sec in s.sections {
            body.appendLE(UInt32(sec.layer))
            body.appendLE(sec.kind.rawValue)
            body.appendLE(UInt64(sec.values.count))
            sec.values.withUnsafeBufferPointer { body.append(Data(buffer: $0)) }
        }
        return signed(body)
    }

    /// Appends the trailing SHA-256 of `body`. Internal so tests can forge
    /// headers and still exercise the semantic refusals.
    static func signed(_ body: Data) -> Data {
        var out = body
        out.append(contentsOf: Array(SHA256.hash(data: body)))
        return out
    }

    // MARK: - Reading

    /// Decodes, verifies against the restoring model, checks the position
    /// against its capacity, and returns a fresh, unbound state.
    static func restore(
        _ data: Data, identity: ModelIdentity, config: QwenConfig, contextCapacity: Int
    ) throws -> QwenCPUModel.DecodeState {
        let snapshot = try decode(data)
        try verify(snapshot, identity: identity, config: config)
        guard snapshot.position <= contextCapacity else {
            throw ConversationStateError.capacityExceeded(position: snapshot.position, capacity: contextCapacity)
        }
        let state = QwenCPUModel.DecodeState()
        state.apply(snapshot)
        return state
    }

    /// Parses and structurally validates a file: magic, version, length,
    /// digest, header fields, and a section table whose every entry fits
    /// the header's own geometry and position. Model identity, geometry,
    /// rotary split and RoPE base are compared separately by `verify`.
    static func decode(_ input: Data) throws -> Snapshot {
        // Reader indexes from 0; a slice handed in from another Data would not.
        let data = input.startIndex == 0 ? input : Data(input)
        guard data.count >= magic.count, Array(data.prefix(magic.count)) == magic else {
            throw ConversationStateError.badMagic
        }
        guard data.count >= headerBytes else { throw ConversationStateError.truncated }
        var r = Reader(data: data, offset: magic.count)
        let version = r.u32()
        guard version == formatVersion else {
            throw ConversationStateError.unsupportedVersion(found: version, supported: formatVersion)
        }
        let declaredHeader = r.u32()
        guard declaredHeader == UInt32(headerBytes) else {
            throw ConversationStateError.corrupt("header length \(declaredHeader)")
        }
        let dtype = r.u32()
        let scheme = r.u32()
        let identity = r.bytes(32)
        var geometryValues: [Int] = []
        for _ in Geometry.fieldNames { geometryValues.append(Int(r.u32())) }
        let position = r.u64()
        let sectionCount = r.u32()
        let total = r.u64()
        let rotaryDims = r.u32()
        let writerContextCapacity = r.u32()
        let ropeThetaBits = r.u64()
        guard total >= UInt64(headerBytes + trailerBytes), total <= UInt64(Int.max) else {
            throw ConversationStateError.corrupt("total length \(total)")
        }
        if data.count < Int(total) { throw ConversationStateError.truncated }
        if data.count > Int(total) { throw ConversationStateError.corrupt("trailing bytes") }

        let bodyEnd = data.count - trailerBytes
        let digest = Array(SHA256.hash(data: data.subdata(in: 0..<bodyEnd)))
        guard digest == Array(data.suffix(trailerBytes)) else {
            throw ConversationStateError.corrupt("digest mismatch")
        }
        // Digest verified: the rest of the bytes are what the writer wrote,
        // so any inconsistency below is the writer's, not transport damage.
        guard dtype == dtypeFloat32 else { throw ConversationStateError.dtypeMismatch(found: dtype) }
        // Range rule before any arithmetic on header fields (see Geometry.fieldCeiling).
        let inRange = 1...Geometry.fieldCeiling
        guard geometryValues.allSatisfy({ inRange.contains($0) }) else {
            throw ConversationStateError.corrupt("geometry field out of range")
        }
        // Only compared, never used in arithmetic: positive is enough.
        guard rotaryDims > 0, writerContextCapacity > 0 else {
            throw ConversationStateError.corrupt("non-positive rotary dims or writer capacity")
        }
        guard sectionCount <= UInt32(Geometry.fieldCeiling) else {
            throw ConversationStateError.corrupt("section count \(sectionCount)")
        }
        let geometry = Geometry(values: geometryValues)
        guard position <= UInt64(Int.max / 4) else {
            throw ConversationStateError.corrupt("position \(position)")
        }

        r = Reader(data: data, offset: headerBytes)
        var sections: [Section] = []
        var seen = Set<SectionKey>()
        for _ in 0..<sectionCount {
            guard r.offset + 16 <= bodyEnd else { throw ConversationStateError.corrupt("section table") }
            let layer = Int(r.u32())
            let kindRaw = r.u32()
            let count = r.u64()
            guard let kind = Kind(rawValue: kindRaw) else {
                throw ConversationStateError.corrupt("section kind \(kindRaw)")
            }
            let expected: Int
            do {
                expected = try expectedCount(layer: layer, kind: kind, geometry: geometry, position: Int(position))
            } catch let problem as ShapeProblem {
                throw ConversationStateError.corrupt(problem.message)
            }
            guard count == UInt64(expected) else {
                throw ConversationStateError.corrupt(
                    "layer \(layer) \(kind) holds \(count) floats, geometry says \(expected)")
            }
            // Bytes remaining, divided — never `expected * 4`, which could overflow.
            guard expected <= (bodyEnd - r.offset) / 4 else { throw ConversationStateError.corrupt("section data") }
            guard seen.insert(SectionKey(layer: layer, kind: kind)).inserted else {
                throw ConversationStateError.corrupt("duplicate section layer \(layer) \(kind)")
            }
            sections.append(Section(layer: layer, kind: kind, values: r.floats(expected)))
        }
        guard r.offset == bodyEnd else { throw ConversationStateError.corrupt("unread section bytes") }
        if let missing = missingSection(in: sections, geometry: geometry, position: Int(position)) {
            throw ConversationStateError.corrupt(missing)
        }
        return Snapshot(identityScheme: scheme, identity: identity, geometry: geometry,
                        rotaryDims: Int(rotaryDims), writerContextCapacity: Int(writerContextCapacity),
                        ropeThetaBits: ropeThetaBits, position: Int(position), sections: sections)
    }

    /// The semantic refusals: same checkpoint, same state geometry, same
    /// rotary split and RoPE base (the K rows are cached post-rotation, so a
    /// K section is only meaningful under the frequencies that produced it).
    static func verify(_ s: Snapshot, identity: ModelIdentity, config: QwenConfig) throws {
        guard s.identityScheme == identity.scheme, s.identity == identity.digest else {
            throw ConversationStateError.modelMismatch(
                expected: identity.hex, found: s.identity.map { String(format: "%02x", $0) }.joined())
        }
        let geometry = Geometry(config: config)
        for (i, name) in Geometry.fieldNames.enumerated() where s.geometry.values[i] != geometry.values[i] {
            throw ConversationStateError.geometryMismatch(
                field: name, expected: geometry.values[i], found: s.geometry.values[i])
        }
        guard s.rotaryDims == config.rotaryDims else {
            throw ConversationStateError.geometryMismatch(
                field: "rotaryDims", expected: config.rotaryDims, found: s.rotaryDims)
        }
        guard s.ropeThetaBits == config.ropeTheta.bitPattern else {
            throw ConversationStateError.ropeThetaMismatch(
                expected: config.ropeTheta, found: Double(bitPattern: s.ropeThetaBits))
        }
    }

    private struct Reader {
        let data: Data
        var offset: Int
        mutating func u32() -> UInt32 {
            defer { offset += 4 }
            return data.subdata(in: offset..<offset + 4).withUnsafeBytes {
                UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self))
            }
        }
        mutating func u64() -> UInt64 {
            defer { offset += 8 }
            return data.subdata(in: offset..<offset + 8).withUnsafeBytes {
                UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))
            }
        }
        mutating func bytes(_ n: Int) -> [UInt8] {
            defer { offset += n }
            return Array(data.subdata(in: offset..<offset + n))
        }
        mutating func floats(_ n: Int) -> [Float] {
            defer { offset += 4 * n }
            return data.subdata(in: offset..<offset + 4 * n).withUnsafeBytes { raw in
                [Float](unsafeUninitializedCapacity: n) { buf, filled in
                    raw.copyBytes(to: buf)
                    filled = n
                }
            }
        }
    }
}

/// What a snapshot names as its model. Cheap to compute (no weight bytes
/// are read) and as strong as the container allows:
///  - scheme 1, container with hashes.json: SHA-256 over config.json and
///    hashes.json — the per-file SHA-256 manifest verify_container.py
///    --write-hashes writes and StreamingInstaller verifies against,
///    { "files": { "<relative path>": "<hex sha256>" } }; a manifest that
///    lists every file (the script always does) pins every weight byte.
///    hashes.json counts only when it decodes as that shape and lists at
///    least one file, the installer's own test before it trusts the file;
///    anything else is treated as absent, so a stray or broken file falls
///    back to scheme 2 instead of pinning config.json plus some bytes.
///  - scheme 2, plain checkpoint (or a container without hashes.json):
///    SHA-256 over config.json, manifest.json if present, and each
///    safetensors shard's name, header JSON, and byte size. This pins the
///    architecture, tensor layout, dtypes, and sizes, not the weight values.
struct ModelIdentity: Equatable {
    let scheme: UInt32
    let digest: [UInt8]
    var hex: String { digest.map { String(format: "%02x", $0) }.joined() }

    static func compute(modelDir: URL, shards: [URL]) throws -> ModelIdentity {
        var hasher = SHA256()
        func feed(_ label: String, _ data: Data) {
            hasher.update(data: Data(label.utf8))
            var n = UInt64(data.count).littleEndian
            withUnsafeBytes(of: &n) { hasher.update(bufferPointer: $0) }
            hasher.update(data: data)
        }
        feed("config.json", try Data(contentsOf: modelDir.appendingPathComponent("config.json")))
        let hashes = modelDir.appendingPathComponent("hashes.json")
        if let hashBytes = try? Data(contentsOf: hashes),
           let manifest = try? JSONDecoder().decode(StreamingInstaller.HashesFile.self, from: hashBytes),
           !manifest.files.isEmpty {
            // The decode only selects the scheme; the digest is over the
            // bytes as written, so the same table re-encoded is another
            // identity.
            feed("hashes.json", hashBytes)
            return ModelIdentity(scheme: 1, digest: Array(hasher.finalize()))
        }
        if let manifest = try? Data(contentsOf: modelDir.appendingPathComponent("manifest.json")) {
            feed("manifest.json", manifest)
        }
        for url in shards.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            let lenBytes = try handle.read(upToCount: 8) ?? Data()
            guard lenBytes.count == 8 else { throw Checkpoint.Error.badShape(url.lastPathComponent) }
            let headerLen = lenBytes.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
            let header = try handle.read(upToCount: Int(headerLen)) ?? Data()
            let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? UInt64) ?? 0
            var sizeLE = size.littleEndian
            feed(url.lastPathComponent, header)
            withUnsafeBytes(of: &sizeLE) { hasher.update(bufferPointer: $0) }
        }
        return ModelIdentity(scheme: 2, digest: Array(hasher.finalize()))
    }
}

// MARK: - DecodeState <-> sections

extension QwenCPUModel.DecodeState {
    /// The state's fields as snapshot sections, one per (layer, kind), in
    /// layer order. Callers on the Metal fast path capture the bound
    /// state's GPU recurrence first (the CPU fields are otherwise stale).
    func snapshotSections() -> [ConversationState.Section] {
        var sections: [ConversationState.Section] = []
        for layer in Set(kv.keys).union(convTail.keys).union(deltaState.keys).sorted() {
            if let rows = kv[layer] {
                sections.append(.init(layer: layer, kind: .k, values: rows.k))
                sections.append(.init(layer: layer, kind: .v, values: rows.v))
            }
            if let tail = convTail[layer] {
                sections.append(.init(layer: layer, kind: .convTail, values: tail))
            }
            if let state = deltaState[layer] {
                sections.append(.init(layer: layer, kind: .deltaState, values: state))
            }
        }
        return sections
    }

    /// Fills a fresh state from a decoded, verified snapshot.
    func apply(_ snapshot: ConversationState.Snapshot) {
        precondition(position == 0 && kv.isEmpty && convTail.isEmpty && deltaState.isEmpty,
                     "apply(_:) expects a fresh DecodeState")
        for section in snapshot.sections {
            switch section.kind {
            case .k: kv[section.layer, default: (k: [], v: [])].k = section.values
            case .v: kv[section.layer, default: (k: [], v: [])].v = section.values
            case .convTail: convTail[section.layer] = section.values
            case .deltaState: deltaState[section.layer] = section.values
            }
        }
        position = snapshot.position
    }
}

// MARK: - Persistence

/// A model whose decode states can be written out and reopened later, in
/// this process or another, on either engine. The bytes are a versioned
/// snapshot that names the model they came from; a restore into a different
/// model, format version, geometry, dtype, or an engine whose context
/// capacity the position exceeds is refused rather than continued from
/// foreign state. Typed on `QwenCPUModel.DecodeState`, the state
/// `InferenceModel.step` takes today; it follows that type wherever the
/// protocol takes it.
public protocol PersistableInferenceModel: InferenceModel {
    /// Serializes `state`, which stays live and unchanged. Refuses, as
    /// `ConversationStateError.inconsistentState`, a state that could not be
    /// restored: its last step threw after it had begun mutating it, or
    /// another engine instance holds it bound.
    func snapshot(of state: QwenCPUModel.DecodeState) throws -> Data
    /// A new state holding the snapshot's contents, ready to continue from
    /// its position on this model.
    func restoreState(from data: Data) throws -> QwenCPUModel.DecodeState
}

extension QwenCPUModel: PersistableInferenceModel {
    /// Model identity for snapshots: config plus hashes.json or the shard
    /// headers, never weight bytes, so it is cheap enough to recompute.
    var stateIdentity: ModelIdentity {
        get throws { try ModelIdentity.compute(modelDir: ckpt.dir, shards: ckpt.shardURLs) }
    }

    public func snapshot(of state: DecodeState) throws -> Data {
        try ConversationState.snapshot(
            of: state, engine: self, identity: try stateIdentity, config: config,
            writerContextCapacity: contextCapacity)
    }

    public func restoreState(from data: Data) throws -> DecodeState {
        try ConversationState.restore(
            data, identity: try stateIdentity, config: config, contextCapacity: contextCapacity)
    }
}

extension QwenMetalModel: PersistableInferenceModel {
    /// Model identity for snapshots: config plus hashes.json or the shard
    /// headers, never weight bytes, so it is cheap enough to recompute.
    var stateIdentity: ModelIdentity {
        get throws { try ModelIdentity.compute(modelDir: ckpt.dir, shards: ckpt.shardURLs) }
    }

    /// Serializes `state`. If it is the bound state its conv history and
    /// delta recurrence are read back from the GPU buffers first (no command
    /// buffer is in flight between steps: every step ends with commitAndWait);
    /// the KV sections come from the state's mirror, which appendKVMirror
    /// copies from the GPU rows after every attention layer (see
    /// ConversationStateTests.metalSnapshotMatchesDirectGPUReadback for the
    /// bitwise proof of both). The state stays bound and live.
    public func snapshot(of state: QwenCPUModel.DecodeState) throws -> Data {
        if let bound = boundState, bound === state, state.position > 0 {
            captureRecurrentState(into: state)
        }
        return try ConversationState.snapshot(
            of: state, engine: self, identity: try stateIdentity, config: config,
            writerContextCapacity: contextCapacity)
    }

    /// The restored state is unbound; its first step loads it into the GPU
    /// buffers through bindState.
    public func restoreState(from data: Data) throws -> QwenCPUModel.DecodeState {
        try ConversationState.restore(
            data, identity: try stateIdentity, config: config, contextCapacity: contextCapacity)
    }
}

private extension Data {
    mutating func appendLE(_ v: UInt32) {
        var le = v.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }
    mutating func appendLE(_ v: UInt64) {
        var le = v.littleEndian
        Swift.withUnsafeBytes(of: &le) { append(contentsOf: $0) }
    }
}
