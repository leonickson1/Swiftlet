import Foundation
import Metal

/// Metal device/queue/pipeline owner for the GPU runtime. The convenience
/// methods here are blocking (dispatch + wait) and copy in/out; the streaming
/// runtime will manage persistent buffers itself.
public final class MetalEngine {
    public enum Error: Swift.Error {
        case noDevice
        case kernelMissing(String)
    }

    public let device: MTLDevice
    public let queue: MTLCommandQueue
    let library: MTLLibrary
    private var pipelines: [String: MTLComputePipelineState] = [:]

    /// A/B switch: SWIFTLET_NO_FAST_GEMV=1 forces the scalar gemv kernel.
    static let fastGemvEnabled =
        ProcessInfo.processInfo.environment["SWIFTLET_NO_FAST_GEMV"] != "1"

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue()
        else { throw Error.noDevice }
        self.device = device
        self.queue = queue
        // Runtime-compiled from bundled source (TurboFieldfare's pattern):
        // plain `swift build` doesn't produce a metallib, and this keeps
        // shader edits a rebuild away with no Xcode step. Shipped as .txt so
        // xcodebuild (iOS) doesn't require the Metal build toolchain.
        guard let url = Bundle.module.url(forResource: "Kernels.metal", withExtension: "txt") else {
            throw Error.kernelMissing("Kernels.metal.txt resource")
        }
        let source = try String(contentsOf: url, encoding: .utf8)
        library = try device.makeLibrary(source: source, options: nil)
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        if let p = pipelines[name] { return p }
        guard let fn = library.makeFunction(name: name) else { throw Error.kernelMissing(name) }
        let p = try device.makeComputePipelineState(function: fn)
        pipelines[name] = p
        return p
    }

    // MARK: - Counter sampling probe (S3b follow-up)

    /// What this device's Metal counter machinery can actually sample —
    /// probed at runtime, never assumed. Capabilities come straight from
    /// `supportsCounterSampling`; `stageBoundarySampleValid` is the result of
    /// really running a sample pass, so "yes" means counters were exercised
    /// on this device, not just advertised.
    public struct CounterSamplingSupport: Equatable, Sendable {
        public let deviceName: String
        /// The MTLCommonCounterSet.timestamp counter set exists on the device.
        public let hasTimestampCounterSet: Bool
        public let atStageBoundary: Bool
        public let atDispatchBoundary: Bool
        public let atDrawBoundary: Bool
        public let atBlitBoundary: Bool
        public let atTileDispatchBoundary: Bool
        /// Functional check: one trivial compute pass ran with stage-boundary
        /// (encoder start/end) timestamp samples attached and both resolved
        /// to a monotone, non-error pair. nil when stage-boundary sampling or
        /// the timestamp set is unavailable, so the pass was never attempted.
        public let stageBoundarySampleValid: Bool?

        public var summary: String {
            func yn(_ b: Bool) -> String { b ? "yes" : "no" }
            let stageProbe: String
            switch stageBoundarySampleValid {
            case .some(true): stageProbe = "(sample resolved)"
            case .some(false): stageProbe = "(sample did NOT resolve)"
            case .none: stageProbe = "(not probed)"
            }
            return "device=\(deviceName) timestampSet=\(yn(hasTimestampCounterSet))"
                + " stage=\(yn(atStageBoundary))\(stageProbe)"
                + " dispatch=\(yn(atDispatchBoundary))"
                + " draw=\(yn(atDrawBoundary))"
                + " blit=\(yn(atBlitBoundary))"
                + " tileDispatch=\(yn(atTileDispatchBoundary))"
        }
    }

    private var counterSamplingCache: CounterSamplingSupport?

    /// The device's timestamp counter set, if it exposes one.
    var timestampCounterSet: MTLCounterSet? {
        device.counterSets?.first {
            $0.name.caseInsensitiveCompare(MTLCommonCounterSet.timestamp.rawValue) == .orderedSame
        }
    }

    /// Probes counter-sampling capabilities once and caches the verdict.
    /// Runs a real stage-boundary sample pass when the device advertises one,
    /// so the report is evidence rather than a table of claims.
    public func probeCounterSampling() -> CounterSamplingSupport {
        if let cached = counterSamplingCache { return cached }
        let tsSet = timestampCounterSet
        let stage = device.supportsCounterSampling(.atStageBoundary)
        var stageValid: Bool?
        if stage, let tsSet {
            stageValid = runStageBoundaryTimestampProbe(counterSet: tsSet)
        }
        let support = CounterSamplingSupport(
            deviceName: device.name,
            hasTimestampCounterSet: tsSet != nil,
            atStageBoundary: stage,
            atDispatchBoundary: device.supportsCounterSampling(.atDispatchBoundary),
            atDrawBoundary: device.supportsCounterSampling(.atDrawBoundary),
            atBlitBoundary: device.supportsCounterSampling(.atBlitBoundary),
            atTileDispatchBoundary: device.supportsCounterSampling(.atTileDispatchBoundary),
            stageBoundarySampleValid: stageValid
        )
        counterSamplingCache = support
        return support
    }

    /// One tiny silu_mul dispatch in its own encoder with encoder start/end
    /// timestamp samples attached — the boundary Apple GPUs support. True
    /// when both samples resolve to a monotone, non-error pair. Deliberately
    /// bypasses `dispatchThreads` so the probe never pollutes step counters.
    private func runStageBoundaryTimestampProbe(counterSet: MTLCounterSet) -> Bool {
        let desc = MTLCounterSampleBufferDescriptor()
        desc.counterSet = counterSet
        desc.storageMode = .shared
        desc.sampleCount = 2
        guard let sampleBuffer = try? device.makeCounterSampleBuffer(descriptor: desc),
              let scratch = device.makeBuffer(length: 64, options: .storageModeShared),
              let pipe = try? pipeline("silu_mul"),
              let cb = queue.makeCommandBuffer()
        else { return false }
        let pass = MTLComputePassDescriptor()
        guard let attachment = pass.sampleBufferAttachments[0] else { return false }
        attachment.sampleBuffer = sampleBuffer
        attachment.startOfEncoderSampleIndex = 0
        attachment.endOfEncoderSampleIndex = 1
        guard let enc = cb.makeComputeCommandEncoder(descriptor: pass) else { return false }
        enc.setComputePipelineState(pipe)
        var p = SIMD4<UInt32>(4, 0, 4, 8)
        enc.setBuffer(scratch, offset: 0, index: 0)
        enc.setBuffer(scratch, offset: 0, index: 1)
        enc.setBuffer(scratch, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 3)
        enc.dispatchThreads(
            MTLSize(width: 4, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 4, height: 1, depth: 1)
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        guard cb.status == .completed,
              let data = try? sampleBuffer.resolveCounterRange(0..<2),
              data.count >= 2 * MemoryLayout<MTLCounterResultTimestamp>.stride
        else { return false }
        let stamps = data.withUnsafeBytes { raw -> (UInt64, UInt64) in
            let t = raw.bindMemory(to: MTLCounterResultTimestamp.self)
            return (t[0].timestamp, t[1].timestamp)
        }
        return stamps.0 != 0 && stamps.1 != 0
            && stamps.0 != .max && stamps.1 != .max
            && stamps.1 >= stamps.0
    }

    struct GemvParams {
        var wOff: UInt64 = 0   // byte offsets, 64-bit (shards exceed 4 GB)
        var sOff: UInt64 = 0
        var bOff: UInt64 = 0
        var outDim: UInt32
        var inDim: UInt32
        var groupSize: UInt32
        var bits: UInt32
        var scalesType: UInt32
        var yOff: UInt32 = 0
        var xOff: UInt32 = 0
    }

    struct GemvPlainParams {
        var wOff: UInt64 = 0
        var outDim: UInt32
        var inDim: UInt32
        var dtype: UInt32
        var yOff: UInt32 = 0
        var xOff: UInt32 = 0
    }

    public enum ScalesType: UInt32 {
        case f32 = 0, f16 = 1, bf16 = 2

        public init?(dtype: String) {
            switch dtype {
            case "F32": self = .f32
            case "F16": self = .f16
            case "BF16": self = .bf16
            default: return nil
            }
        }
    }

    func makeBuffer<T>(_ array: [T]) -> MTLBuffer {
        array.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! }
    }

    func makeBuffer(_ data: Data) -> MTLBuffer {
        data.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: max($0.count, 1))! }
    }

    // MARK: - Encoder-level API (persistent-buffer runtime path)

    /// Set only while QwenMetalModel.step is active. Keeping the hook here
    /// counts actual dispatch encodes across both fast and fallback paths.
    var computeDispatchObserver: (() -> Void)?

    func dispatchThreads(
        _ enc: MTLComputeCommandEncoder,
        threads: MTLSize,
        threadsPerThreadgroup: MTLSize
    ) {
        enc.dispatchThreads(threads, threadsPerThreadgroup: threadsPerThreadgroup)
        computeDispatchObserver?()
    }

    /// Encode one GEMV over a shard-resident linear. `wExtra`/`sExtra`/`bExtra`
    /// advance into stacked tensors (expert slicing), in the same units as the
    /// descriptor offsets. Output rows land at `y[yOff...]`.
    func encodeGemv(
        _ enc: MTLComputeCommandEncoder,
        _ lin: MetalShardStore.GPULinear,
        rows: Int? = nil,
        wExtra: Int = 0, sExtra: Int = 0, bExtra: Int = 0,
        x: MTLBuffer, xOff: Int = 0, y: MTLBuffer, yOff: Int
    ) throws {
        let outDim = rows ?? lin.outDim
        if lin.isQuantized {
            // Cooperative fast path: one simdgroup per row, vectorized loads.
            // Needs 4-byte-aligned rows (qpack blobs and resident buffers are).
            // SWIFTLET_NO_FAST_GEMV=1 forces the scalar kernel (A/B debugging).
            let aligned = (lin.wOff + wExtra) % 4 == 0
            let fastName: String? = !Self.fastGemvEnabled || !aligned ? nil
                : lin.bits == 4 && lin.groupSize % 8 == 0 ? "gemv_affine_fast"
                : lin.bits == 8 && lin.groupSize % 4 == 0 ? "gemv_affine_fast8"
                : nil
            if let fastName {
                let pipe = try pipeline(fastName)
                enc.setComputePipelineState(pipe)
                var p = GemvParams(
                    wOff: UInt64(lin.wOff + wExtra), sOff: UInt64(lin.sOff + sExtra),
                    bOff: UInt64(lin.bOff + bExtra),
                    outDim: UInt32(outDim), inDim: UInt32(lin.inDim),
                    groupSize: UInt32(lin.groupSize), bits: UInt32(lin.bits),
                    scalesType: lin.scalesType, yOff: UInt32(yOff), xOff: UInt32(xOff)
                )
                enc.setBuffer(x, offset: 0, index: 0)
                enc.setBuffer(lin.wBuffer, offset: 0, index: 1)
                enc.setBuffer(lin.sBuffer, offset: 0, index: 2)
                enc.setBuffer(lin.bBuffer, offset: 0, index: 3)
                enc.setBuffer(y, offset: 0, index: 4)
                enc.setBytes(&p, length: MemoryLayout<GemvParams>.stride, index: 5)
                dispatchThreads(
                    enc, threads: MTLSize(width: 32, height: outDim, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                )
                return
            }
            let pipe = try pipeline("gemv_affine")
            enc.setComputePipelineState(pipe)
            var p = GemvParams(
                wOff: UInt64(lin.wOff + wExtra), sOff: UInt64(lin.sOff + sExtra),
                bOff: UInt64(lin.bOff + bExtra),
                outDim: UInt32(outDim), inDim: UInt32(lin.inDim),
                groupSize: UInt32(lin.groupSize), bits: UInt32(lin.bits),
                scalesType: lin.scalesType, yOff: UInt32(yOff), xOff: UInt32(xOff)
            )
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(lin.wBuffer, offset: 0, index: 1)
            enc.setBuffer(lin.sBuffer, offset: 0, index: 2)
            enc.setBuffer(lin.bBuffer, offset: 0, index: 3)
            enc.setBuffer(y, offset: 0, index: 4)
            enc.setBytes(&p, length: MemoryLayout<GemvParams>.stride, index: 5)
            dispatchThreads(
                enc, threads: MTLSize(width: outDim, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(64, outDim), height: 1, depth: 1)
            )
        } else {
            let pipe = try pipeline("gemv_plain")
            enc.setComputePipelineState(pipe)
            var p = GemvPlainParams(
                wOff: UInt64(lin.wOff + wExtra),
                outDim: UInt32(outDim), inDim: UInt32(lin.inDim),
                dtype: lin.plainDtype, yOff: UInt32(yOff), xOff: UInt32(xOff)
            )
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(lin.wBuffer, offset: 0, index: 1)
            enc.setBuffer(y, offset: 0, index: 2)
            enc.setBytes(&p, length: MemoryLayout<GemvPlainParams>.stride, index: 3)
            dispatchThreads(
                enc, threads: MTLSize(width: outDim, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(64, outDim), height: 1, depth: 1)
            )
        }
    }

    /// The setBytes offset-table ceiling for one batched GEMV dispatch:
    /// 512 uint2 entries stay well under Metal's 4 KB setBytes limit. Longer
    /// tables split into further dispatches.
    static let gemvBatchSlotLimit = 512

    /// Encode the same linear applied to many token slots (S1b token
    /// batching): the grid gains a slot dimension and each slice reads its
    /// x/y float offsets from a small table, so one dispatch replaces the
    /// per-token `encodeGemv` sequence. The per-row arithmetic is the
    /// single-token kernel's verbatim, making every output row bitwise
    /// identical to the unbatched encode. A single slot delegates to
    /// `encodeGemv` so degenerate chunks keep the decode-shaped encode
    /// stream; empty slot lists encode nothing.
    func encodeGemvBatch(
        _ enc: MTLComputeCommandEncoder,
        _ lin: MetalShardStore.GPULinear,
        rows: Int? = nil,
        wExtra: Int = 0, sExtra: Int = 0, bExtra: Int = 0,
        x: MTLBuffer, y: MTLBuffer,
        slots: [(xOff: Int, yOff: Int)]
    ) throws {
        guard let first = slots.first else { return }
        if slots.count == 1 {
            try encodeGemv(enc, lin, rows: rows, wExtra: wExtra, sExtra: sExtra, bExtra: bExtra,
                           x: x, xOff: first.xOff, y: y, yOff: first.yOff)
            return
        }
        let outDim = rows ?? lin.outDim
        var start = 0
        while start < slots.count {
            let end = min(start + Self.gemvBatchSlotLimit, slots.count)
            let table = slots[start..<end].map {
                SIMD2<UInt32>(UInt32($0.xOff), UInt32($0.yOff))
            }
            let n = end - start
            if lin.isQuantized {
                // Same kernel selection as encodeGemv: cooperative fast path
                // when rows are 4-byte aligned, scalar fallback otherwise.
                let aligned = (lin.wOff + wExtra) % 4 == 0
                let fastName: String? = !Self.fastGemvEnabled || !aligned ? nil
                    : lin.bits == 4 && lin.groupSize % 8 == 0 ? "gemv_affine_fast_batch"
                    : lin.bits == 8 && lin.groupSize % 4 == 0 ? "gemv_affine_fast8_batch"
                    : nil
                enc.setComputePipelineState(try pipeline(fastName ?? "gemv_affine_batch"))
                var p = GemvParams(
                    wOff: UInt64(lin.wOff + wExtra), sOff: UInt64(lin.sOff + sExtra),
                    bOff: UInt64(lin.bOff + bExtra),
                    outDim: UInt32(outDim), inDim: UInt32(lin.inDim),
                    groupSize: UInt32(lin.groupSize), bits: UInt32(lin.bits),
                    scalesType: lin.scalesType
                )
                enc.setBuffer(x, offset: 0, index: 0)
                enc.setBuffer(lin.wBuffer, offset: 0, index: 1)
                enc.setBuffer(lin.sBuffer, offset: 0, index: 2)
                enc.setBuffer(lin.bBuffer, offset: 0, index: 3)
                enc.setBuffer(y, offset: 0, index: 4)
                enc.setBytes(&p, length: MemoryLayout<GemvParams>.stride, index: 5)
                table.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 6) }
                if fastName != nil {
                    dispatchThreads(
                        enc, threads: MTLSize(width: 32, height: outDim, depth: n),
                        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
                    )
                } else {
                    dispatchThreads(
                        enc, threads: MTLSize(width: outDim, height: n, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(64, outDim), height: 1, depth: 1)
                    )
                }
            } else {
                enc.setComputePipelineState(try pipeline("gemv_plain_batch"))
                var p = GemvPlainParams(
                    wOff: UInt64(lin.wOff + wExtra),
                    outDim: UInt32(outDim), inDim: UInt32(lin.inDim),
                    dtype: lin.plainDtype
                )
                enc.setBuffer(x, offset: 0, index: 0)
                enc.setBuffer(lin.wBuffer, offset: 0, index: 1)
                enc.setBuffer(y, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<GemvPlainParams>.stride, index: 3)
                table.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 4) }
                dispatchThreads(
                    enc, threads: MTLSize(width: outDim, height: n, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(64, outDim), height: 1, depth: 1)
                )
            }
            start = end
        }
    }

    // MARK: - S1b non-GEMV batch twins
    //
    // The chunked prefill's per-token glue destinations are slot-regular
    // (token t's rows sit exactly one stride after token t-1's), so these
    // twins batch via a token grid dimension plus stride parameters instead
    // of the GEMV offset table. Every kernel body is the single-token
    // kernel's verbatim, so batched rows are bitwise the per-token values.
    // Callers keep single-token chunks on the legacy per-token encodes so
    // the degenerate chunk stays byte-identical to the decode stream.

    struct NormCopyBatchParams {
        var dim: UInt32
        var eps: Float
        var dstOff: UInt32
        var dstStride: UInt32
        var hStride: UInt32
    }

    /// norm_copy over `tokens` hidden rows: row t reads `h` (from
    /// `hByteOffset`) at t*hStride floats and writes dst[dstOff + t*dstStride...].
    func encodeNormCopyBatch(
        _ enc: MTLComputeCommandEncoder,
        h: MTLBuffer, hByteOffset: Int, hStride: Int,
        dst: MTLBuffer, weight: MTLBuffer,
        dim: Int, eps: Float, dstOff: Int, dstStride: Int, tokens: Int
    ) throws {
        enc.setComputePipelineState(try pipeline("norm_copy_batch"))
        var p = NormCopyBatchParams(
            dim: UInt32(dim), eps: eps, dstOff: UInt32(dstOff),
            dstStride: UInt32(dstStride), hStride: UInt32(hStride)
        )
        enc.setBuffer(h, offset: hByteOffset, index: 0)
        enc.setBuffer(dst, offset: 0, index: 1)
        enc.setBuffer(weight, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<NormCopyBatchParams>.stride, index: 3)
        dispatchThreads(
            enc, threads: MTLSize(width: 32, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    /// silu_mul over `tokens` slots: slot t adds t*stride to every offset.
    func encodeSiluMulBatch(
        _ enc: MTLComputeCommandEncoder,
        buf: MTLBuffer, count: Int, gOff: Int, uOff: Int, dstOff: Int,
        stride: Int, tokens: Int
    ) throws {
        enc.setComputePipelineState(try pipeline("silu_mul_batch"))
        var po = SIMD4<UInt32>(UInt32(count), UInt32(gOff), UInt32(uOff), UInt32(dstOff))
        var s = UInt32(stride)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBuffer(buf, offset: 0, index: 1)
        enc.setBuffer(buf, offset: 0, index: 2)
        enc.setBytes(&po, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 3)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.stride, index: 4)
        dispatchThreads(
            enc, threads: MTLSize(width: count, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, count), height: 1, depth: 1)
        )
    }

    /// add_inplace over `tokens` hidden rows:
    /// h[t*hStride + i] += r[rOff + t*rStride + i].
    func encodeAddInplaceBatch(
        _ enc: MTLComputeCommandEncoder,
        h: MTLBuffer, hStride: Int, r: MTLBuffer, rOff: Int, rStride: Int,
        count: Int, tokens: Int
    ) throws {
        enc.setComputePipelineState(try pipeline("add_inplace_batch"))
        var po = SIMD4<UInt32>(
            UInt32(count), UInt32(rOff), UInt32(hStride), UInt32(rStride))
        enc.setBuffer(h, offset: 0, index: 0)
        enc.setBuffer(r, offset: 0, index: 1)
        enc.setBytes(&po, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 2)
        dispatchThreads(
            enc, threads: MTLSize(width: count, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, count), height: 1, depth: 1)
        )
    }

    struct AccumBatchParams {
        var D: UInt32
        var K: UInt32
        var dBase: UInt32
        var shOff: UInt32
        var gateOff: UInt32
        var stride: UInt32
        var hStride: UInt32
    }

    /// weighted_accum over `tokens`: token t reads its expert/shared rows at
    /// t*stride, its own K weights at weights[t*K...], and accumulates into
    /// hidden row t*hStride. The K-expert accumulation order stays inside
    /// the kernel, exactly the single-token loop. weights.count == tokens*K.
    func encodeWeightedAccumBatch(
        _ enc: MTLComputeCommandEncoder,
        h: MTLBuffer, hStride: Int, data: MTLBuffer,
        count: Int, k: Int, dBase: Int, shOff: Int, gateOff: Int, stride: Int,
        weights: [Float], tokens: Int
    ) throws {
        precondition(weights.count == tokens * k, "token-major weights table")
        enc.setComputePipelineState(try pipeline("weighted_accum_batch"))
        var p = AccumBatchParams(
            D: UInt32(count), K: UInt32(k), dBase: UInt32(dBase),
            shOff: UInt32(shOff), gateOff: UInt32(gateOff),
            stride: UInt32(stride), hStride: UInt32(hStride)
        )
        enc.setBuffer(h, offset: 0, index: 0)
        enc.setBuffer(data, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<AccumBatchParams>.stride, index: 2)
        weights.withUnsafeBufferPointer {
            enc.setBytes($0.baseAddress!, length: max(1, $0.count) * 4, index: 3)
        }
        dispatchThreads(
            enc, threads: MTLSize(width: count, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, count), height: 1, depth: 1)
        )
    }

    struct DeintBatchParams {
        var nk: UInt32
        var dk: UInt32
        var rep: UInt32
        var dv: UInt32
        var keyDim: UInt32
        var srcOff: UInt32
        var baOff: UInt32
        var qkvOff: UInt32
        var zOff: UInt32
        var bOff: UInt32
        var aOff: UInt32
        var stride: UInt32
    }

    /// deinterleave_qkvz over `tokens` slots: slot t adds t*stride to every
    /// source and destination offset.
    func encodeDeinterleaveBatch(
        _ enc: MTLComputeCommandEncoder, data: MTLBuffer,
        nk: Int, dk: Int, rep: Int, dv: Int, keyDim: Int,
        srcOff: Int, baOff: Int, qkvOff: Int, zOff: Int, bOff: Int, aOff: Int,
        stride: Int, tokens: Int
    ) throws {
        enc.setComputePipelineState(try pipeline("deinterleave_qkvz_batch"))
        var p = DeintBatchParams(
            nk: UInt32(nk), dk: UInt32(dk), rep: UInt32(rep), dv: UInt32(dv),
            keyDim: UInt32(keyDim), srcOff: UInt32(srcOff), baOff: UInt32(baOff),
            qkvOff: UInt32(qkvOff), zOff: UInt32(zOff), bOff: UInt32(bOff),
            aOff: UInt32(aOff), stride: UInt32(stride)
        )
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBytes(&p, length: MemoryLayout<DeintBatchParams>.stride, index: 1)
        dispatchThreads(
            enc, threads: MTLSize(width: nk, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, nk), height: 1, depth: 1)
        )
    }

    struct AttnPrepBatchParams {
        var heads: UInt32
        var headDim: UInt32
        var rot: UInt32
        var eps: Float
        var theta: Float
        var position: UInt32
        var srcOff: UInt32
        var dstOff: UInt32
        var stride: UInt32
    }

    /// attn_q_prep over `tokens`: token t preps inside its own slot stride at
    /// position basePosition + t. One simdgroup per (head, token).
    func encodeAttnQPrepBatch(
        _ enc: MTLComputeCommandEncoder, data: MTLBuffer, weight: MTLBuffer,
        heads: Int, headDim: Int, rotaryDims: Int, eps: Float, ropeTheta: Float,
        basePosition: Int, srcOff: Int, dstOff: Int, slotStride: Int, tokens: Int
    ) throws {
        enc.setComputePipelineState(try pipeline("attn_q_prep_batch"))
        var p = AttnPrepBatchParams(
            heads: UInt32(heads), headDim: UInt32(headDim), rot: UInt32(rotaryDims),
            eps: eps, theta: ropeTheta, position: UInt32(basePosition),
            srcOff: UInt32(srcOff), dstOff: UInt32(dstOff), stride: UInt32(slotStride)
        )
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(weight, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<AttnPrepBatchParams>.stride, index: 2)
        dispatchThreads(
            enc, threads: MTLSize(width: 32, height: heads, depth: tokens),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    struct AttnKVBatchParams {
        var kvHeads: UInt32
        var headDim: UInt32
        var rot: UInt32
        var eps: Float
        var theta: Float
        var position: UInt32
        var kSrcOff: UInt32
        var vSrcOff: UInt32
        var stride: UInt32
    }

    /// attn_kv_append over `tokens`: token t appends at cache row index
    /// basePosition + t — disjoint rows per token, so batching cannot
    /// reorder any write. One simdgroup per (kv head, token).
    func encodeAttnKVAppendBatch(
        _ enc: MTLComputeCommandEncoder, data: MTLBuffer, weight: MTLBuffer,
        kCache: MTLBuffer, vCache: MTLBuffer,
        kvHeads: Int, headDim: Int, rotaryDims: Int, eps: Float, ropeTheta: Float,
        basePosition: Int, kSrcOff: Int, vSrcOff: Int, slotStride: Int, tokens: Int
    ) throws {
        enc.setComputePipelineState(try pipeline("attn_kv_append_batch"))
        var p = AttnKVBatchParams(
            kvHeads: UInt32(kvHeads), headDim: UInt32(headDim), rot: UInt32(rotaryDims),
            eps: eps, theta: ropeTheta, position: UInt32(basePosition),
            kSrcOff: UInt32(kSrcOff), vSrcOff: UInt32(vSrcOff), stride: UInt32(slotStride)
        )
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(weight, offset: 0, index: 1)
        enc.setBuffer(kCache, offset: 0, index: 2)
        enc.setBuffer(vCache, offset: 0, index: 3)
        enc.setBytes(&p, length: MemoryLayout<AttnKVBatchParams>.stride, index: 4)
        dispatchThreads(
            enc, threads: MTLSize(width: 32, height: kvHeads, depth: tokens),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    struct AttnDecodeBatchParams {
        var heads: UInt32
        var kvHeads: UInt32
        var headDim: UInt32
        var position: UInt32
        var qOff: UInt32
        var qoutOff: UInt32
        var outOff: UInt32
        var stride: UInt32
    }

    /// attn_decode_gqa over `tokens`: token t attends inside its own slot
    /// stride over cache rows [0, basePosition + t + 1) — the causal kvLen
    /// derived in-kernel from the token grid. Callers must barrier between
    /// the chunk's KV appends and this dispatch, because the youngest token
    /// reads every appended row. One threadgroup of `attnDecodeSimdgroups`
    /// simdgroups per (head, token), tiled across the KV rows.
    func encodeAttnDecodeBatch(
        _ enc: MTLComputeCommandEncoder, data: MTLBuffer,
        kCache: MTLBuffer, vCache: MTLBuffer,
        heads: Int, kvHeads: Int, headDim: Int, basePosition: Int,
        qOff: Int, qoutOff: Int, outOff: Int, slotStride: Int, tokens: Int
    ) throws {
        precondition(headDim <= 256, "attn_decode_gqa accumulators cover headDim <= 256")
        enc.setComputePipelineState(try pipeline("attn_decode_gqa_batch"))
        var p = AttnDecodeBatchParams(
            heads: UInt32(heads), kvHeads: UInt32(kvHeads), headDim: UInt32(headDim),
            position: UInt32(basePosition), qOff: UInt32(qOff),
            qoutOff: UInt32(qoutOff), outOff: UInt32(outOff), stride: UInt32(slotStride)
        )
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(kCache, offset: 0, index: 1)
        enc.setBuffer(vCache, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<AttnDecodeBatchParams>.stride, index: 3)
        let width = 32 * Self.attnDecodeSimdgroups
        dispatchThreads(
            enc, threads: MTLSize(width: width, height: heads, depth: tokens),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
    }

    struct DeltaPreBatchParams {
        var nv: UInt32
        var aOff: UInt32
        var bOff: UInt32
        var outOff: UInt32
        var stride: UInt32
    }

    /// delta_pre over `tokens` slots: slot t adds t*slotStride to every
    /// offset (g at outOff, beta at outOff + nv; aLog/dtBias shared).
    func encodeDeltaPreBatch(
        _ enc: MTLComputeCommandEncoder, data: MTLBuffer,
        aLog: MTLBuffer, dtBias: MTLBuffer,
        nv: Int, aOff: Int, bOff: Int, outOff: Int, slotStride: Int, tokens: Int
    ) throws {
        enc.setComputePipelineState(try pipeline("delta_pre_batch"))
        var p = DeltaPreBatchParams(
            nv: UInt32(nv), aOff: UInt32(aOff), bOff: UInt32(bOff),
            outOff: UInt32(outOff), stride: UInt32(slotStride)
        )
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(aLog, offset: 0, index: 1)
        enc.setBuffer(dtBias, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<DeltaPreBatchParams>.stride, index: 3)
        dispatchThreads(
            enc, threads: MTLSize(width: nv, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, nv), height: 1, depth: 1)
        )
    }

    struct NormBatchParams {
        var rows: UInt32
        var dim: UInt32
        var eps: Float
        var hasWeight: UInt32
        var scale: Float
        var off: UInt32
        var stride: UInt32
    }

    /// rmsnorm_rows over `tokens` slots, in place: slot t norms `rows` rows
    /// of `dim` at off + t*slotStride. Optional weight, optional extra scale
    /// (the DeltaNet q/k norms are weightless, scaled 1/dk and 1/sqrt(dk)).
    /// One simdgroup per (row, token).
    func encodeRMSNormRowsBatch(
        _ enc: MTLComputeCommandEncoder, data: MTLBuffer, weight: MTLBuffer?,
        off: Int, rows: Int, dim: Int, scale: Float, eps: Float,
        slotStride: Int, tokens: Int
    ) throws {
        enc.setComputePipelineState(try pipeline("rmsnorm_rows_batch"))
        var p = NormBatchParams(
            rows: UInt32(rows), dim: UInt32(dim), eps: eps,
            hasWeight: weight != nil ? 1 : 0, scale: scale,
            off: UInt32(off), stride: UInt32(slotStride)
        )
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(weight ?? data, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<NormBatchParams>.stride, index: 2)
        dispatchThreads(
            enc, threads: MTLSize(width: 32, height: rows, depth: tokens),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    struct GatedNormBatchParams {
        var nv: UInt32
        var dv: UInt32
        var eps: Float
        var yOff: UInt32
        var zOff: UInt32
        var outOff: UInt32
        var yStride: UInt32
        var zStride: UInt32
        var outStride: UInt32
    }

    /// gated_norm_mul over `tokens` with per-stream strides: the chunked
    /// recurrence reads y rows from the contiguous [T, row] staging block
    /// while z and the output stay slot-strided. One simdgroup per
    /// (v-head, token).
    func encodeGatedNormMulBatch(
        _ enc: MTLComputeCommandEncoder, data: MTLBuffer, weight: MTLBuffer,
        nv: Int, dv: Int, eps: Float,
        yOff: Int, yStride: Int, zOff: Int, zStride: Int,
        outOff: Int, outStride: Int, tokens: Int
    ) throws {
        enc.setComputePipelineState(try pipeline("gated_norm_mul_batch"))
        var p = GatedNormBatchParams(
            nv: UInt32(nv), dv: UInt32(dv), eps: eps,
            yOff: UInt32(yOff), zOff: UInt32(zOff), outOff: UInt32(outOff),
            yStride: UInt32(yStride), zStride: UInt32(zStride),
            outStride: UInt32(outStride)
        )
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(weight, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<GatedNormBatchParams>.stride, index: 2)
        dispatchThreads(
            enc, threads: MTLSize(width: 32, height: nv, depth: tokens),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    struct DeltaGatherParams {
        var keyDim: UInt32
        var valueDim: UInt32
        var nv: UInt32
        var slotStride: UInt32
        var convOff: UInt32
        var gbOff: UInt32
        var qOut: UInt32
        var kOut: UInt32
        var vOut: UInt32
        var gOut: UInt32
        var bOut: UInt32
    }

    /// Packs every chunk token's normed q/k/v conv rows and g/beta pairs
    /// from slot-strided regions into contiguous [tokens, row] blocks — the
    /// layout gated_delta_step's T-step scan expects. Pure element copies;
    /// staging cannot perturb numerics.
    func encodeDeltaGather(
        _ enc: MTLComputeCommandEncoder, data: MTLBuffer,
        keyDim: Int, valueDim: Int, nv: Int,
        slotStride: Int, convOff: Int, gbOff: Int,
        qOut: Int, kOut: Int, vOut: Int, gOut: Int, bOut: Int, tokens: Int
    ) throws {
        enc.setComputePipelineState(try pipeline("delta_gather"))
        var p = DeltaGatherParams(
            keyDim: UInt32(keyDim), valueDim: UInt32(valueDim), nv: UInt32(nv),
            slotStride: UInt32(slotStride), convOff: UInt32(convOff), gbOff: UInt32(gbOff),
            qOut: UInt32(qOut), kOut: UInt32(kOut), vOut: UInt32(vOut),
            gOut: UInt32(gOut), bOut: UInt32(bOut)
        )
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBytes(&p, length: MemoryLayout<DeltaGatherParams>.stride, index: 1)
        let width = 2 * keyDim + valueDim + 2 * nv
        dispatchThreads(
            enc, threads: MTLSize(width: width, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, width), height: 1, depth: 1)
        )
    }

    struct ConvScanParams {
        var convDim: UInt32
        var K: UInt32
        var inOff: UInt32
        var outOff: UInt32
        var stride: UInt32
        var T: UInt32
    }

    /// conv_scan over `tokens` slots: T sequential conv_step iterations in
    /// one dispatch, one thread per channel — token t's qkv row read at
    /// inOff + t*slotStride, its conv+silu row written at
    /// outOff + t*slotStride, and the layer's K-1 history rows carried
    /// across tokens in registers, written back once at the end. Bitwise
    /// the chained per-token conv_step dispatches. K <= 8.
    func encodeConvScan(
        _ enc: MTLComputeCommandEncoder, data: MTLBuffer, hist: MTLBuffer,
        weight: MTLBuffer, convDim: Int, K: Int,
        inOff: Int, outOff: Int, slotStride: Int, tokens: Int
    ) throws {
        precondition((2...8).contains(K), "conv_scan history registers cover 2 <= K <= 8")
        enc.setComputePipelineState(try pipeline("conv_scan"))
        var p = ConvScanParams(
            convDim: UInt32(convDim), K: UInt32(K), inOff: UInt32(inOff),
            outOff: UInt32(outOff), stride: UInt32(slotStride), T: UInt32(tokens)
        )
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(hist, offset: 0, index: 1)
        enc.setBuffer(weight, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<ConvScanParams>.stride, index: 3)
        dispatchThreads(
            enc, threads: MTLSize(width: convDim, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, convDim), height: 1, depth: 1)
        )
    }

    func encodeSiluMul(
        _ enc: MTLComputeCommandEncoder,
        buf: MTLBuffer, count: Int, gOff: Int, uOff: Int, dstOff: Int
    ) throws {
        let pipe = try pipeline("silu_mul")
        enc.setComputePipelineState(pipe)
        var p = SIMD4<UInt32>(UInt32(count), UInt32(gOff), UInt32(uOff), UInt32(dstOff))
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBuffer(buf, offset: 0, index: 1)
        enc.setBuffer(buf, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 3)
        dispatchThreads(
            enc, threads: MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: min(64, count), height: 1, depth: 1)
        )
    }

    /// Blocking quantized GEMV: y = W x with MLX affine layout.
    public func gemvQuantized(
        x: [Float], packed: Data, scales: Data, biases: Data,
        outDim: Int, inDim: Int, groupSize: Int, bits: Int, scalesType: ScalesType,
        useFast: Bool = false
    ) throws -> [Float] {
        let pipe = try pipeline(
            useFast ? (bits == 4 ? "gemv_affine_fast" : "gemv_affine_fast8") : "gemv_affine")
        var params = GemvParams(
            outDim: UInt32(outDim), inDim: UInt32(inDim),
            groupSize: UInt32(groupSize), bits: UInt32(bits),
            scalesType: scalesType.rawValue, yOff: 0, xOff: 0
        )
        let yBuf = device.makeBuffer(length: outDim * 4)!

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        enc.setBuffer(makeBuffer(x), offset: 0, index: 0)
        enc.setBuffer(makeBuffer(packed), offset: 0, index: 1)
        enc.setBuffer(makeBuffer(scales), offset: 0, index: 2)
        enc.setBuffer(makeBuffer(biases), offset: 0, index: 3)
        enc.setBuffer(yBuf, offset: 0, index: 4)
        enc.setBytes(&params, length: MemoryLayout<GemvParams>.stride, index: 5)
        if useFast {
            enc.dispatchThreads(
                MTLSize(width: 32, height: outDim, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
            )
        } else {
            let tg = min(pipe.maxTotalThreadsPerThreadgroup, 64)
            enc.dispatchThreads(
                MTLSize(width: outDim, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1)
            )
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        return Array(UnsafeBufferPointer(
            start: yBuf.contents().bindMemory(to: Float.self, capacity: outDim), count: outDim
        ))
    }

    // MARK: - Full-attention kernels (S2)

    struct AttnPrepParams {
        var heads: UInt32
        var headDim: UInt32
        var rot: UInt32
        var eps: Float
        var theta: Float
        var position: UInt32
        var srcOff: UInt32
        var dstOff: UInt32
    }

    struct AttnKVParams {
        var kvHeads: UInt32
        var headDim: UInt32
        var rot: UInt32
        var eps: Float
        var theta: Float
        var position: UInt32
        var kSrcOff: UInt32
        var vSrcOff: UInt32
    }

    struct AttnDecodeParams {
        var heads: UInt32
        var kvHeads: UInt32
        var headDim: UInt32
        var kvLen: UInt32
        var qOff: UInt32
        var qoutOff: UInt32
        var outOff: UInt32
    }

    /// Extract each head's query from the interleaved q|gate projection at
    /// `srcOff` (floats, per head [q(hd) | gate(hd)]), RMSNorm it with
    /// `weight`, apply partial RoPE for `position`, and write flat rows at
    /// `dstOff`. The gate half stays untouched at `srcOff` for the attention
    /// kernel. One simdgroup per head.
    func encodeAttnQPrep(
        _ enc: MTLComputeCommandEncoder, data: MTLBuffer, weight: MTLBuffer,
        heads: Int, headDim: Int, rotaryDims: Int, eps: Float, ropeTheta: Float,
        position: Int, srcOff: Int, dstOff: Int
    ) throws {
        enc.setComputePipelineState(try pipeline("attn_q_prep"))
        var p = AttnPrepParams(
            heads: UInt32(heads), headDim: UInt32(headDim), rot: UInt32(rotaryDims),
            eps: eps, theta: ropeTheta, position: UInt32(position),
            srcOff: UInt32(srcOff), dstOff: UInt32(dstOff)
        )
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(weight, offset: 0, index: 1)
        enc.setBytes(&p, length: MemoryLayout<AttnPrepParams>.stride, index: 2)
        dispatchThreads(
            enc, threads: MTLSize(width: 32, height: heads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    /// Append one position's K/V rows to the caches at row index `position`
    /// (layout [position][kvHead][headDim], matching the CPU reference): K is
    /// RMSNormed with `weight` and roped, V copies verbatim. One simdgroup
    /// per kv head.
    func encodeAttnKVAppend(
        _ enc: MTLComputeCommandEncoder, data: MTLBuffer, weight: MTLBuffer,
        kCache: MTLBuffer, vCache: MTLBuffer,
        kvHeads: Int, headDim: Int, rotaryDims: Int, eps: Float, ropeTheta: Float,
        position: Int, kSrcOff: Int, vSrcOff: Int
    ) throws {
        enc.setComputePipelineState(try pipeline("attn_kv_append"))
        var p = AttnKVParams(
            kvHeads: UInt32(kvHeads), headDim: UInt32(headDim), rot: UInt32(rotaryDims),
            eps: eps, theta: ropeTheta, position: UInt32(position),
            kSrcOff: UInt32(kSrcOff), vSrcOff: UInt32(vSrcOff)
        )
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(weight, offset: 0, index: 1)
        enc.setBuffer(kCache, offset: 0, index: 2)
        enc.setBuffer(vCache, offset: 0, index: 3)
        enc.setBytes(&p, length: MemoryLayout<AttnKVParams>.stride, index: 4)
        dispatchThreads(
            enc, threads: MTLSize(width: 32, height: kvHeads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    /// Simdgroups cooperating on one (head, token) in the causal attends,
    /// each scanning a strided subset of the KV rows with its own online
    /// softmax, merged in-threadgroup. Must match ATTN_DECODE_SG in
    /// Kernels.metal.txt.
    static let attnDecodeSimdgroups = 8

    /// Gated causal GQA attention for one query token over cache rows
    /// [0, kvLen): out = sigmoid(gate) * softmax(qK^T/sqrt(hd)) V, with the
    /// online-softmax running max/sum in f32. Prepped q at `qOff`, the gate
    /// half read from the raw projection at `qoutOff`, gated context written
    /// to `outOff`. One threadgroup of `attnDecodeSimdgroups` simdgroups
    /// per query head, tiled across the KV rows; headDim <= 256.
    func encodeAttnDecode(
        _ enc: MTLComputeCommandEncoder, data: MTLBuffer,
        kCache: MTLBuffer, vCache: MTLBuffer,
        heads: Int, kvHeads: Int, headDim: Int, kvLen: Int,
        qOff: Int, qoutOff: Int, outOff: Int
    ) throws {
        precondition(headDim <= 256, "attn_decode_gqa accumulators cover headDim <= 256")
        enc.setComputePipelineState(try pipeline("attn_decode_gqa"))
        var p = AttnDecodeParams(
            heads: UInt32(heads), kvHeads: UInt32(kvHeads), headDim: UInt32(headDim),
            kvLen: UInt32(kvLen), qOff: UInt32(qOff), qoutOff: UInt32(qoutOff),
            outOff: UInt32(outOff)
        )
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(kCache, offset: 0, index: 1)
        enc.setBuffer(vCache, offset: 0, index: 2)
        enc.setBytes(&p, length: MemoryLayout<AttnDecodeParams>.stride, index: 3)
        let width = 32 * Self.attnDecodeSimdgroups
        dispatchThreads(
            enc, threads: MTLSize(width: width, height: heads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
    }

    struct DeltaParams {
        var T: UInt32
        var Hk: UInt32
        var Hv: UInt32
        var Dk: UInt32
        var Dv: UInt32
    }

    /// Blocking gated-delta recurrence over T steps (B = 1). Returns y
    /// [T, Hv, Dv] and the updated state [Hv, Dv, Dk].
    public func gatedDeltaStep(
        q: [Float], k: [Float], v: [Float], g: [Float], beta: [Float],
        state: [Float], T: Int, Hk: Int, Hv: Int, Dk: Int, Dv: Int
    ) throws -> (y: [Float], state: [Float]) {
        precondition(Dk % 32 == 0 && Dk <= 256 && Dv % 4 == 0)
        let pipe = try pipeline("gated_delta_step")
        var params = DeltaParams(T: UInt32(T), Hk: UInt32(Hk), Hv: UInt32(Hv), Dk: UInt32(Dk), Dv: UInt32(Dv))
        let yBuf = device.makeBuffer(length: T * Hv * Dv * 4)!
        let stateOut = device.makeBuffer(length: state.count * 4)!

        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        for (i, arr) in [q, k, v, g, beta, state].enumerated() {
            enc.setBuffer(makeBuffer(arr), offset: 0, index: i)
        }
        enc.setBuffer(yBuf, offset: 0, index: 6)
        enc.setBuffer(stateOut, offset: 0, index: 7)
        enc.setBytes(&params, length: MemoryLayout<DeltaParams>.stride, index: 8)
        enc.dispatchThreads(
            MTLSize(width: 32, height: Dv, depth: Hv),
            threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let y = Array(UnsafeBufferPointer(
            start: yBuf.contents().bindMemory(to: Float.self, capacity: T * Hv * Dv), count: T * Hv * Dv
        ))
        let s = Array(UnsafeBufferPointer(
            start: stateOut.contents().bindMemory(to: Float.self, capacity: state.count), count: state.count
        ))
        return (y, s)
    }
}
