import Foundation
import Metal

/// Memory-maps checkpoint shard files and exposes them as GPU-visible
/// MTLBuffers (unified memory, zero copy). Weights are never decompressed:
/// kernels compute directly on the quantized bytes, and the OS page cache
/// serves as the expert cache — hot pages stay resident, cold ones fault in
/// from the internal SSD.
final class MetalShardStore {
    enum Error: Swift.Error {
        case mapFailed(String)
        case badAlignment(String)
    }

    private let device: MTLDevice
    private var mappings: [URL: MTLBuffer] = [:]
    /// Bytes every `residentLinear` buffer occupies, at Metal's allocated
    /// size (never below the requested length). The resident dense trunk the
    /// expert-cache budget is priced against.
    private(set) var residentBytes = 0
    private(set) var residentLinearCount = 0

    init(device: MTLDevice) {
        self.device = device
    }

    private func chargeResident(_ buffer: MTLBuffer) {
        residentBytes += max(buffer.length, buffer.allocatedSize)
        residentLinearCount += 1
    }

    func buffer(for url: URL) throws -> MTLBuffer {
        if let m = mappings[url] { return m }
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw Error.mapFailed(url.lastPathComponent) }
        defer { close(fd) }
        var st = stat()
        fstat(fd, &st)
        let pageSize = Int(getpagesize())
        let mapLen = (Int(st.st_size) + pageSize - 1) / pageSize * pageSize
        guard let ptr = mmap(nil, mapLen, PROT_READ, MAP_PRIVATE, fd, 0),
              ptr != MAP_FAILED
        else { throw Error.mapFailed(url.lastPathComponent) }
        guard let buf = device.makeBuffer(
            bytesNoCopy: ptr, length: mapLen, options: .storageModeShared,
            deallocator: { p, len in munmap(p, len) }
        ) else {
            munmap(ptr, mapLen)
            throw Error.mapFailed("MTLBuffer for \(url.lastPathComponent)")
        }
        mappings[url] = buf
        return buf
    }

    /// GPU-executable descriptor for one linear module (quantized or plain).
    /// Offsets are in BYTES (kernels are byte-addressed: safetensors headers
    /// aren't padded, so tensors sit at arbitrary offsets).
    struct GPULinear {
        var wBuffer: MTLBuffer
        var sBuffer: MTLBuffer   // == wBuffer unless the module straddles shards
        var bBuffer: MTLBuffer
        var outDim: Int
        var inDim: Int
        var isQuantized: Bool
        var groupSize: Int = 0
        var bits: Int = 0
        var scalesType: UInt32 = 0
        var wOff: Int = 0        // bytes
        var sOff: Int = 0
        var bOff: Int = 0
        var plainDtype: UInt32 = 0
    }

    /// RESIDENT variant: copies the module's bytes into one owned MTLBuffer
    /// (weight | scales | biases). Used for dense weights so they can never be
    /// evicted or paged, unlike the mmap-backed variant below.
    func residentLinear(ckpt: Checkpoint, path: String) throws -> GPULinear {
        if ckpt.isQuantized(path), let spec = ckpt.quantSpec(for: path) {
            // Allocate the destination buffer once, copy each tensor straight
            // from the mapped file: no intermediate Data (matters on iOS —
            // transient copies of large modules invite jetsam during load).
            let wInfo = try ckpt.tensorLocation(path + ".weight").info
            let sInfo = try ckpt.tensorLocation(path + ".scales").info
            let bInfo = try ckpt.tensorLocation(path + ".biases").info
            let wLen = wInfo.byteRange.count
            let sLen = sInfo.byteRange.count
            let bLen = bInfo.byteRange.count
            guard let buf = device.makeBuffer(length: wLen + sLen + bLen, options: .storageModeShared) else {
                throw Error.mapFailed(path)
            }
            try ckpt.withRawTensor(path + ".weight") { _, bytes in
                buf.contents().copyMemory(from: bytes.baseAddress!, byteCount: wLen)
            }
            try ckpt.withRawTensor(path + ".scales") { _, bytes in
                buf.contents().advanced(by: wLen).copyMemory(from: bytes.baseAddress!, byteCount: sLen)
            }
            try ckpt.withRawTensor(path + ".biases") { _, bytes in
                buf.contents().advanced(by: wLen + sLen).copyMemory(from: bytes.baseAddress!, byteCount: bLen)
            }
            chargeResident(buf)
            let perWord = 32 / spec.bits
            return GPULinear(
                wBuffer: buf, sBuffer: buf, bBuffer: buf,
                outDim: wInfo.shape.dropLast().reduce(1, *),
                inDim: wInfo.shape.last! * perWord,
                isQuantized: true,
                groupSize: spec.groupSize, bits: spec.bits,
                scalesType: MetalEngine.ScalesType(dtype: sInfo.dtype)?.rawValue ?? 2,
                wOff: 0, sOff: wLen, bOff: wLen + sLen
            )
        }
        let wInfo = try ckpt.tensorLocation(path + ".weight").info
        let wLen = wInfo.byteRange.count
        guard let buf = device.makeBuffer(length: wLen, options: .storageModeShared) else {
            throw Error.mapFailed(path)
        }
        try ckpt.withRawTensor(path + ".weight") { _, bytes in
            buf.contents().copyMemory(from: bytes.baseAddress!, byteCount: wLen)
        }
        chargeResident(buf)
        let dtype: UInt32 = wInfo.dtype == "F32" ? 0 : (wInfo.dtype == "F16" ? 1 : 2)
        return GPULinear(
            wBuffer: buf, sBuffer: buf, bBuffer: buf,
            outDim: wInfo.shape.dropLast().reduce(1, *),
            inDim: wInfo.shape.last!,
            isQuantized: false,
            wOff: 0,
            plainDtype: dtype
        )
    }

    func gpuLinear(ckpt: Checkpoint, path: String) throws -> GPULinear {
        let (wURL, wByteOff, wInfo) = try ckpt.tensorLocation(path + ".weight")
        let wBuf = try buffer(for: wURL)

        if ckpt.isQuantized(path), let spec = ckpt.quantSpec(for: path) {
            let (sURL, sByteOff, sInfo) = try ckpt.tensorLocation(path + ".scales")
            let (bURL, bByteOff, _) = try ckpt.tensorLocation(path + ".biases")
            let perWord = 32 / spec.bits
            return GPULinear(
                wBuffer: wBuf,
                sBuffer: try buffer(for: sURL),
                bBuffer: try buffer(for: bURL),
                outDim: wInfo.shape.dropLast().reduce(1, *),
                inDim: wInfo.shape.last! * perWord,
                isQuantized: true,
                groupSize: spec.groupSize,
                bits: spec.bits,
                scalesType: MetalEngine.ScalesType(dtype: sInfo.dtype)?.rawValue ?? 2,
                wOff: wByteOff,
                sOff: sByteOff,
                bOff: bByteOff
            )
        }

        let dtype: UInt32 = wInfo.dtype == "F32" ? 0 : (wInfo.dtype == "F16" ? 1 : 2)
        return GPULinear(
            wBuffer: wBuf, sBuffer: wBuf, bBuffer: wBuf,
            outDim: wInfo.shape.dropLast().reduce(1, *),
            inDim: wInfo.shape.last!,
            isQuantized: false,
            wOff: wByteOff,
            plainDtype: dtype
        )
    }
}
