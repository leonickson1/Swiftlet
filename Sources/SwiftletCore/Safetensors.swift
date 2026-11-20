import Foundation

/// Minimal safetensors reader: 8-byte little-endian header length, JSON header
/// mapping tensor name -> { dtype, shape, data_offsets }, then raw data.
public struct SafetensorsFile {
    public struct TensorInfo: Sendable {
        public let dtype: String
        public let shape: [Int]
        public let byteRange: Range<Int>
    }

    private let data: Data
    private let dataStart: Int
    public private(set) var tensors: [String: TensorInfo] = [:]

    public enum Error: Swift.Error, CustomStringConvertible {
        case malformedHeader(String)
        case missingTensor(String)
        case unsupportedDtype(String, tensor: String)

        public var description: String {
            switch self {
            case .malformedHeader(let why): return "malformed safetensors header: \(why)"
            case .missingTensor(let name): return "missing tensor \(name)"
            case .unsupportedDtype(let dtype, let name): return "unsupported dtype \(dtype) for tensor \(name)"
            }
        }
    }

    public init(url: URL) throws {
        // External drives occasionally interrupt the initial map (EINTR);
        // retry briefly before giving up.
        var mapped: Data? = nil
        var lastError: Swift.Error = Error.malformedHeader("could not read header")
        for attempt in 0..<4 {
            do {
                mapped = try Data(contentsOf: url, options: .mappedIfSafe)
                break
            } catch {
                lastError = error
                usleep(useconds_t(100_000 * (attempt + 1)))
            }
        }
        guard let mapped else { throw lastError }
        data = mapped
        guard data.count >= 8 else { throw Error.malformedHeader("header length prefix is shorter than 8 bytes") }
        let length = data.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: 0, as: UInt64.self)) }
        guard let headerLen = Int(exactly: length) else {
            throw Error.malformedHeader("header length \(length) does not fit Int")
        }
        guard headerLen <= data.count - 8 else {
            throw Error.malformedHeader("header length \(length) exceeds the file's \(data.count - 8) remaining bytes")
        }
        dataStart = 8 + headerLen
        let headerJSON = try JSONSerialization.jsonObject(with: data.subdata(in: 8..<dataStart))
        guard let header = headerJSON as? [String: Any] else {
            throw Error.malformedHeader("header JSON is not an object")
        }
        for (name, value) in header where name != "__metadata__" {
            tensors[name] = try Self.tensorInfo(value, name: name, dataSize: data.count - dataStart)
        }
    }

    /// Validate untrusted tensor metadata before constructing a range or using
    /// any shape arithmetic. Also used for headers fetched without the payload.
    static func tensorInfo(_ value: Any, name: String, dataSize: Int) throws -> TensorInfo {
        guard let entry = value as? [String: Any],
              let dtype = entry["dtype"] as? String,
              let shape = entry["shape"] as? [Int],
              let offsets = entry["data_offsets"] as? [Int], offsets.count == 2
        else { throw Error.malformedHeader("tensor \(name): invalid dtype, shape or data_offsets") }
        let begin = offsets[0], end = offsets[1]
        guard begin >= 0, end >= begin, end <= dataSize else {
            throw Error.malformedHeader(
                "tensor \(name): data_offsets [\(begin), \(end)] must satisfy 0 <= begin <= end <= \(dataSize)")
        }
        guard shape.allSatisfy({ $0 >= 0 }) else {
            throw Error.malformedHeader("tensor \(name): shape \(shape) has a negative dimension")
        }
        guard let elementSize = bytesPerElement(dtype) else {
            throw Error.malformedHeader("tensor \(name): unsupported dtype \(dtype)")
        }
        var byteCount = elementSize
        if shape.contains(0) {
            byteCount = 0
        } else {
            for dimension in shape {
                let (product, overflow) = byteCount.multipliedReportingOverflow(by: dimension)
                guard !overflow else {
                    throw Error.malformedHeader("tensor \(name): dtype \(dtype) × shape \(shape) byte count overflows Int")
                }
                byteCount = product
            }
        }
        guard end - begin == byteCount else {
            throw Error.malformedHeader(
                "tensor \(name): data_offsets span \(end - begin) bytes, but dtype \(dtype) × shape \(shape) requires \(byteCount)")
        }
        return TensorInfo(dtype: dtype, shape: shape, byteRange: begin..<end)
    }

    public func info(_ name: String) throws -> TensorInfo {
        guard let t = tensors[name] else { throw Error.missingTensor(name) }
        return t
    }

    func rawBytes(_ t: TensorInfo) -> Data {
        data.subdata(in: (dataStart + t.byteRange.lowerBound)..<(dataStart + t.byteRange.upperBound))
    }

    /// A sub-range of one tensor's stored bytes, viewed in place in the
    /// mapped file -- no copy of the tensor. The data section can start at
    /// any byte offset (headers are not padded), so callers must load
    /// through `loadUnaligned` or memcpy, never `bindMemory`.
    func withTensorBytes<T>(
        _ t: TensorInfo, byteRange: Range<Int>, _ body: (UnsafeRawBufferPointer) throws -> T
    ) rethrows -> T {
        precondition(byteRange.lowerBound >= 0 && byteRange.upperBound <= t.byteRange.count,
                     "tensor byte range out of bounds")
        let lo = dataStart + t.byteRange.lowerBound + byteRange.lowerBound
        return try data.withUnsafeBytes { raw in
            try body(UnsafeRawBufferPointer(rebasing: raw[lo..<(lo + byteRange.count)]))
        }
    }

    /// Raw stored bytes of a tensor (no dtype conversion), with its info.
    public func raw(_ name: String) throws -> (info: TensorInfo, bytes: Data) {
        let t = try info(name)
        return (t, rawBytes(t))
    }

    /// Byte offset of a tensor's data from the start of the FILE (for mmap use).
    public func absoluteOffset(_ name: String) throws -> Int {
        dataStart + (try info(name)).byteRange.lowerBound
    }

    /// Zero-copy access to a tensor's raw bytes (view into the mapped file).
    public func withRawBytes<T>(_ name: String, _ body: (SafetensorsFile.TensorInfo, UnsafeRawBufferPointer) throws -> T) throws -> T {
        let t = try info(name)
        return try data.withUnsafeBytes { raw in
            let slice = UnsafeRawBufferPointer(
                rebasing: raw[(dataStart + t.byteRange.lowerBound)..<(dataStart + t.byteRange.upperBound)]
            )
            return try body(t, slice)
        }
    }

    public static func bytesPerElement(_ dtype: String) -> Int? {
        switch dtype {
        case "F32", "I32", "U32": return 4
        case "F16", "BF16", "I16", "U16": return 2
        case "F64", "I64", "U64": return 8
        case "I8", "U8": return 1
        default: return nil
        }
    }

    /// Serialize tensors into safetensors format (used by the repacker for the
    /// dense/resident weight file).
    public static func write(to url: URL, tensors: [(name: String, dtype: String, shape: [Int], bytes: Data)]) throws {
        var header: [String: Any] = [:]
        var offset = 0
        for t in tensors {
            header[t.name] = [
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": [offset, offset + t.bytes.count],
            ]
            offset += t.bytes.count
        }
        let headerData = try JSONSerialization.data(withJSONObject: header)
        var out = Data()
        var len = UInt64(headerData.count).littleEndian
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(headerData)
        for t in tensors { out.append(t.bytes) }
        try out.write(to: url)
    }

    /// Tensor contents converted to Float32. Supports F32, F16, BF16.
    public func floats(_ name: String) throws -> [Float] {
        let t = try info(name)
        let bytes = rawBytes(t)
        switch t.dtype {
        case "F32":
            return bytes.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        case "F16":
            return bytes.withUnsafeBytes { $0.bindMemory(to: Float16.self).map(Float.init) }
        case "BF16":
            return bytes.withUnsafeBytes {
                $0.bindMemory(to: UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
            }
        default:
            throw Error.unsupportedDtype(t.dtype, tensor: name)
        }
    }

    /// Integer tensor contents. Supports I32, U32, I64.
    public func ints(_ name: String) throws -> [Int] {
        let t = try info(name)
        let bytes = rawBytes(t)
        switch t.dtype {
        case "I32": return bytes.withUnsafeBytes { $0.bindMemory(to: Int32.self).map(Int.init) }
        case "U32": return bytes.withUnsafeBytes { $0.bindMemory(to: UInt32.self).map(Int.init) }
        case "I64": return bytes.withUnsafeBytes { $0.bindMemory(to: Int64.self).map(Int.init) }
        default:
            throw Error.unsupportedDtype(t.dtype, tensor: name)
        }
    }
}
