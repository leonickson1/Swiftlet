import CryptoKit
import Foundation

/// Why a snapshot was refused. Every case is a hard stop: nothing is
/// restored from bytes that fail any check.
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
    /// The elements are not float32.
    case dtypeMismatch(found: UInt32)
}

/// The versioned binary snapshot of a Qwen inference context. See
/// docs/STATE_FORMAT.md for the byte layout; the constants and offsets there
/// are pinned by ConversationStateTests.headerLayoutIsPinned.
///
/// Container: fixed 128-byte little-endian header, a run of float32
/// sections, and a trailing SHA-256 over everything before it. The header
/// carries the format version, dtype, a model identity digest (with the
/// scheme that produced it), the eleven geometry fields that shape the
/// state, the position, the section count, and the total byte length, so
/// a reader can refuse a truncated, foreign, or mis-shaped file before it
/// interprets a single section.
public enum ConversationState {
    public static let formatVersion: UInt32 = 1
    static let magic: [UInt8] = Array("SWLSTATE".utf8)
    static let headerBytes = 128
    static let trailerBytes = 32
    /// Element dtype codes. Only float32 exists in v1.
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
    /// config (experts, MoE sizes, rope) does not change what a context
    /// holds and is covered by the identity digest instead.
    struct Geometry: Equatable {
        static let fieldNames = [
            "numHiddenLayers", "fullAttentionInterval", "numKeyValueHeads", "headDim",
            "linearNumValueHeads", "linearNumKeyHeads", "linearKeyHeadDim",
            "linearValueHeadDim", "linearConvKernelDim", "hiddenSize", "vocabSize",
        ]
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
        var kvRowFloats: Int { values[2] * values[3] }
        var convTailFloats: Int { (values[8] - 1) * convDim }
        var deltaStateFloats: Int { values[4] * values[7] * values[6] }
        private var convDim: Int { 2 * values[5] * values[6] + values[4] * values[7] }
        func isLinearLayer(_ index: Int) -> Bool { (index + 1) % fullAttentionInterval != 0 }
    }

    /// A decoded file: everything the header and sections say, before it
    /// is compared with any model.
    struct Snapshot {
        var identityScheme: UInt32
        var identity: [UInt8]
        var geometry: Geometry
        var position: Int
        var sections: [Section]

        func section(layer: Int, kind: Kind) -> Section? {
            sections.first { $0.layer == layer && $0.kind == kind }
        }
    }

    // MARK: - Encoding

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

    // MARK: - Decoding

    /// Parses and structurally validates a file: magic, version, length,
    /// digest, header fields, and a section table whose every entry fits
    /// the header's own geometry and position. Model identity and geometry
    /// are compared separately by `verify`.
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
        guard geometryValues.allSatisfy({ $0 > 0 }) else {
            throw ConversationStateError.corrupt("non-positive geometry field")
        }
        let geometry = Geometry(values: geometryValues)
        guard position <= UInt64(Int.max / 4) else {
            throw ConversationStateError.corrupt("position \(position)")
        }

        r = Reader(data: data, offset: headerBytes)
        var sections: [Section] = []
        for _ in 0..<sectionCount {
            guard r.offset + 16 <= bodyEnd else { throw ConversationStateError.corrupt("section table") }
            let layer = Int(r.u32())
            let kindRaw = r.u32()
            let count = r.u64()
            guard let kind = Kind(rawValue: kindRaw) else {
                throw ConversationStateError.corrupt("section kind \(kindRaw)")
            }
            guard layer < geometry.numHiddenLayers else {
                throw ConversationStateError.corrupt("section layer \(layer)")
            }
            let expected: UInt64
            switch kind {
            case .k, .v:
                guard !geometry.isLinearLayer(layer) else {
                    throw ConversationStateError.corrupt("KV section on DeltaNet layer \(layer)")
                }
                expected = position * UInt64(geometry.kvRowFloats)
            case .convTail:
                guard geometry.isLinearLayer(layer) else {
                    throw ConversationStateError.corrupt("conv tail on attention layer \(layer)")
                }
                expected = UInt64(geometry.convTailFloats)
            case .deltaState:
                guard geometry.isLinearLayer(layer) else {
                    throw ConversationStateError.corrupt("delta state on attention layer \(layer)")
                }
                expected = UInt64(geometry.deltaStateFloats)
            }
            guard count == expected else {
                throw ConversationStateError.corrupt(
                    "layer \(layer) \(kind) holds \(count) floats, geometry says \(expected)")
            }
            let byteCount = Int(count) * 4
            guard r.offset + byteCount <= bodyEnd else { throw ConversationStateError.corrupt("section data") }
            guard !sections.contains(where: { $0.layer == layer && $0.kind == kind }) else {
                throw ConversationStateError.corrupt("duplicate section layer \(layer) \(kind)")
            }
            sections.append(Section(layer: layer, kind: kind, values: r.floats(Int(count))))
        }
        guard r.offset == bodyEnd else { throw ConversationStateError.corrupt("unread section bytes") }

        // Every layer must be fully present once the sequence has started.
        if position > 0 {
            for li in 0..<geometry.numHiddenLayers {
                let kinds: [Kind] = geometry.isLinearLayer(li) ? [.convTail, .deltaState] : [.k, .v]
                for kind in kinds where !sections.contains(where: { $0.layer == li && $0.kind == kind }) {
                    throw ConversationStateError.corrupt("layer \(li) missing \(kind)")
                }
            }
        }
        return Snapshot(identityScheme: scheme, identity: identity, geometry: geometry,
                        position: Int(position), sections: sections)
    }

    /// The semantic refusals: same checkpoint, same state geometry.
    static func verify(_ s: Snapshot, identity: ModelIdentity, geometry: Geometry) throws {
        guard s.identityScheme == identity.scheme, s.identity == identity.digest else {
            throw ConversationStateError.modelMismatch(
                expected: identity.hex, found: s.identity.map { String(format: "%02x", $0) }.joined())
        }
        for (i, name) in Geometry.fieldNames.enumerated() where s.geometry.values[i] != geometry.values[i] {
            throw ConversationStateError.geometryMismatch(
                field: name, expected: geometry.values[i], found: s.geometry.values[i])
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
///  - scheme 1, qpack container with hashes.json: SHA-256 over config.json
///    and hashes.json, whose per-file SHA-256s pin every weight byte.
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
        if let hashBytes = try? Data(contentsOf: hashes) {
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
