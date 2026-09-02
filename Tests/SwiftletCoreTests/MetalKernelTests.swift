import Foundation
import Metal
import Testing
@testable import SwiftletCore

/// Metal kernels vs exact CPU references on deterministic pseudo-random data.
@Suite struct MetalKernelTests {
    /// Deterministic LCG so failures reproduce.
    struct Rand {
        var s: UInt64
        init(_ seed: UInt64) { s = seed }
        mutating func next32() -> UInt32 {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            return UInt32(truncatingIfNeeded: s >> 32)
        }
        mutating func float() -> Float { Float(next32()) / Float(UInt32.max) * 2 - 1 }
    }

    static func bf16(_ f: Float) -> UInt16 { UInt16(truncatingIfNeeded: f.bitPattern >> 16) }
    static func fromBF16(_ u: UInt16) -> Float { Float(bitPattern: UInt32(u) << 16) }

    @Test(arguments: [4, 8], [false, true])
    func gemvAffineMatchesCPU(bits: Int, useFast: Bool) throws {
        let engine = try MetalEngine()
        let O = 48, I = 128, group = 32
        let perWord = 32 / bits
        var rng = Rand(42)

        let packed = (0..<O * I / perWord).map { _ in rng.next32() }
        let scalesF = (0..<O * I / group).map { _ in rng.float() * 0.1 }
        let biasesF = (0..<O * I / group).map { _ in rng.float() * 0.05 }
        let x = (0..<I).map { _ in rng.float() }

        func cpuReference(scaleAt: (Int) -> Float, biasAt: (Int) -> Float) -> [Float] {
            var y = [Float](repeating: 0, count: O)
            let mask = UInt32((1 << bits) - 1)
            let groups = I / group
            for o in 0..<O {
                var acc: Float = 0
                for i in 0..<I {
                    let word = packed[o * (I / perWord) + i / perWord]
                    let q = Float((word >> (UInt32(bits) * UInt32(i % perWord))) & mask)
                    let g = o * groups + i / group
                    acc += (scaleAt(g) * q + biasAt(g)) * x[i]
                }
                y[o] = acc
            }
            return y
        }

        // f32 scales.
        let yF32 = try engine.gemvQuantized(
            x: x,
            packed: packed.withUnsafeBytes { Data($0) },
            scales: scalesF.withUnsafeBytes { Data($0) },
            biases: biasesF.withUnsafeBytes { Data($0) },
            outDim: O, inDim: I, groupSize: group, bits: bits, scalesType: .f32,
            useFast: useFast
        )
        let refF32 = cpuReference(scaleAt: { scalesF[$0] }, biasAt: { biasesF[$0] })
        for i in 0..<O { #expect(abs(yF32[i] - refF32[i]) < 1e-3, "f32 row \(i): \(yF32[i]) vs \(refF32[i])") }

        // bf16 scales: reference uses the same truncated values.
        let scalesB = scalesF.map(Self.bf16)
        let biasesB = biasesF.map(Self.bf16)
        let yBF16 = try engine.gemvQuantized(
            x: x,
            packed: packed.withUnsafeBytes { Data($0) },
            scales: scalesB.withUnsafeBytes { Data($0) },
            biases: biasesB.withUnsafeBytes { Data($0) },
            outDim: O, inDim: I, groupSize: group, bits: bits, scalesType: .bf16,
            useFast: useFast
        )
        let refBF16 = cpuReference(
            scaleAt: { Self.fromBF16(scalesB[$0]) }, biasAt: { Self.fromBF16(biasesB[$0]) }
        )
        for i in 0..<O { #expect(abs(yBF16[i] - refBF16[i]) < 1e-3, "bf16 row \(i)") }
    }

    @Test func gatedDeltaStepMatchesCPU() throws {
        let engine = try MetalEngine()
        let T = 5, Hk = 2, Hv = 4, Dk = 32, Dv = 8
        var rng = Rand(7)

        let q = (0..<T * Hk * Dk).map { _ in rng.float() }
        let k = (0..<T * Hk * Dk).map { _ in rng.float() }
        let v = (0..<T * Hv * Dv).map { _ in rng.float() }
        let g = (0..<T * Hv).map { _ in abs(rng.float()) * 0.9 }
        let beta = (0..<T * Hv).map { _ in abs(rng.float()) }
        let state0 = (0..<Hv * Dv * Dk).map { _ in rng.float() * 0.1 }

        // CPU reference: references/gated_delta.py ops path.
        var refState = state0
        var refY = [Float](repeating: 0, count: T * Hv * Dv)
        let rep = Hv / Hk
        for t in 0..<T {
            for hv in 0..<Hv {
                let hk = hv / rep
                let qB = t * Hk * Dk + hk * Dk
                let kB = t * Hk * Dk + hk * Dk
                let vB = t * Hv * Dv + hv * Dv
                for dvi in 0..<Dv {
                    let sB = (hv * Dv + dvi) * Dk
                    var kvMem: Float = 0
                    for i in 0..<Dk {
                        refState[sB + i] *= g[t * Hv + hv]
                        kvMem += refState[sB + i] * k[kB + i]
                    }
                    let delta = (v[vB + dvi] - kvMem) * beta[t * Hv + hv]
                    var out: Float = 0
                    for i in 0..<Dk {
                        refState[sB + i] += k[kB + i] * delta
                        out += refState[sB + i] * q[qB + i]
                    }
                    refY[t * Hv * Dv + hv * Dv + dvi] = out
                }
            }
        }

        let (y, state) = try engine.gatedDeltaStep(
            q: q, k: k, v: v, g: g, beta: beta, state: state0,
            T: T, Hk: Hk, Hv: Hv, Dk: Dk, Dv: Dv
        )
        var maxY: Float = 0, maxS: Float = 0
        for i in 0..<refY.count { maxY = max(maxY, abs(y[i] - refY[i])) }
        for i in 0..<refState.count { maxS = max(maxS, abs(state[i] - refState[i])) }
        #expect(maxY < 1e-4, "delta y maxAbsDiff \(maxY)")
        #expect(maxS < 1e-4, "delta state maxAbsDiff \(maxS)")
    }

    // MARK: - S1b token-batched GEMV

    static func quantizedLinear(
        _ engine: MetalEngine, bits: Int, pad: Int, outDim O: Int = 48, inDim I: Int = 128
    ) -> MetalShardStore.GPULinear {
        let group = 32
        let perWord = 32 / bits
        var rng = Rand(UInt64(bits * 100 + pad + 7))
        let packed = (0..<O * I / perWord).map { _ in rng.next32() }
        let scales = (0..<O * I / group).map { _ in rng.float() * 0.1 }
        let biases = (0..<O * I / group).map { _ in rng.float() * 0.05 }
        // One blob: [pad] weights | scales | biases. pad = 2 yields weight
        // rows that are only 2-byte aligned, forcing the scalar kernel.
        var blob = Data(repeating: 0, count: pad)
        packed.withUnsafeBytes { blob.append(contentsOf: $0) }
        let sOff = blob.count
        scales.withUnsafeBytes { blob.append(contentsOf: $0) }
        let bOff = blob.count
        biases.withUnsafeBytes { blob.append(contentsOf: $0) }
        let buf = engine.makeBuffer(blob)
        return MetalShardStore.GPULinear(
            wBuffer: buf, sBuffer: buf, bBuffer: buf, outDim: O, inDim: I,
            isQuantized: true, groupSize: group, bits: bits, scalesType: 0,
            wOff: pad, sOff: sOff, bOff: bOff
        )
    }

    static func plainLinear(
        _ engine: MetalEngine, dtype: UInt32, outDim O: Int = 48, inDim I: Int = 128
    ) -> MetalShardStore.GPULinear {
        var rng = Rand(UInt64(900 + dtype))
        let rows = (0..<O * I).map { _ in rng.float() }
        var blob = Data()
        switch dtype {
        case 0: rows.withUnsafeBytes { blob.append(contentsOf: $0) }
        case 1:
            let h = rows.map { Float16($0) }
            h.withUnsafeBytes { blob.append(contentsOf: $0) }
        default:
            let b = rows.map { Self.bf16($0) }
            b.withUnsafeBytes { blob.append(contentsOf: $0) }
        }
        let buf = engine.makeBuffer(blob)
        return MetalShardStore.GPULinear(
            wBuffer: buf, sBuffer: buf, bBuffer: buf, outDim: O, inDim: I,
            isQuantized: false, plainDtype: dtype
        )
    }

    /// Applies `lin` to `tokens` slots (distinct x/y offsets in shared
    /// buffers) once through per-token encodeGemv dispatches and once through
    /// encodeGemvBatch, requiring bitwise-identical output buffers plus the
    /// mechanical dispatch reduction.
    static func expectBatchMatches(
        _ engine: MetalEngine, _ lin: MetalShardStore.GPULinear, label: String,
        tokens: Int = 3, expectedBatchDispatches: Int = 1
    ) throws {
        let O = lin.outDim, I = lin.inDim
        let xStride = I + 7, yStride = O + 5
        var rng = Rand(4242)
        let x = (0..<tokens * xStride).map { _ in rng.float() }
        let xBuf = engine.makeBuffer(x)
        let slots = (0..<tokens).map { (xOff: $0 * xStride, yOff: $0 * yStride + 3) }
        let yFloats = tokens * yStride + 3

        func run(_ body: (MTLComputeCommandEncoder, MTLBuffer) throws -> Void)
            throws -> (y: [Float], dispatches: Int)
        {
            let yBuf = engine.device.makeBuffer(length: yFloats * 4)!
            memset(yBuf.contents(), 0, yFloats * 4)
            var dispatches = 0
            engine.computeDispatchObserver = { dispatches += 1 }
            defer { engine.computeDispatchObserver = nil }
            let cb = engine.queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            try body(enc, yBuf)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            let y = Array(UnsafeBufferPointer(
                start: yBuf.contents().bindMemory(to: Float.self, capacity: yFloats),
                count: yFloats))
            return (y, dispatches)
        }

        let perToken = try run { enc, yBuf in
            for s in slots {
                try engine.encodeGemv(enc, lin, x: xBuf, xOff: s.xOff, y: yBuf, yOff: s.yOff)
            }
        }
        let batched = try run { enc, yBuf in
            try engine.encodeGemvBatch(enc, lin, x: xBuf, y: yBuf, slots: slots)
        }
        #expect(perToken.dispatches == tokens, "\(label): per-token dispatch count")
        #expect(batched.dispatches == expectedBatchDispatches, "\(label): batched dispatch count")
        #expect(perToken.y.map(\.bitPattern) == batched.y.map(\.bitPattern),
                "\(label): batched GEMV diverges bitwise from per-token dispatches")
    }

    /// Every kernel variant the runtime selects — fast 4/8-bit, the scalar
    /// fallback (2-byte-aligned weights), and each plain dtype — must batch
    /// bitwise-identically, with N token dispatches collapsing to one.
    @Test func gemvBatchMatchesPerTokenBitwise() throws {
        let engine = try MetalEngine()
        try Self.expectBatchMatches(engine, Self.quantizedLinear(engine, bits: 4, pad: 0), label: "q4 fast")
        try Self.expectBatchMatches(engine, Self.quantizedLinear(engine, bits: 8, pad: 0), label: "q8 fast")
        try Self.expectBatchMatches(engine, Self.quantizedLinear(engine, bits: 4, pad: 2), label: "q4 scalar")
        try Self.expectBatchMatches(engine, Self.plainLinear(engine, dtype: 0), label: "plain f32")
        try Self.expectBatchMatches(engine, Self.plainLinear(engine, dtype: 1), label: "plain f16")
        try Self.expectBatchMatches(engine, Self.plainLinear(engine, dtype: 2), label: "plain bf16")
    }

    /// A single slot delegates to the unbatched encode (one dispatch, same
    /// bytes as the decode schedule); a table past the setBytes ceiling
    /// splits into ceil(n / 512) dispatches, still bitwise.
    @Test func gemvBatchRespectsTableLimits() throws {
        let engine = try MetalEngine()
        let lin = Self.quantizedLinear(engine, bits: 4, pad: 0, outDim: 8, inDim: 32)
        try Self.expectBatchMatches(engine, lin, label: "single slot", tokens: 1)
        try Self.expectBatchMatches(
            engine, lin, label: "split table", tokens: 513, expectedBatchDispatches: 2)
    }

    // MARK: - S1b non-GEMV batch twins

    /// Encodes `body` into one command buffer, waits, and returns how many
    /// compute dispatches it encoded (via the engine's dispatch observer).
    static func runEncoded(
        _ engine: MetalEngine, _ body: (MTLComputeCommandEncoder) throws -> Void
    ) throws -> Int {
        var dispatches = 0
        engine.computeDispatchObserver = { dispatches += 1 }
        defer { engine.computeDispatchObserver = nil }
        let cb = engine.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        do {
            try body(enc)
        } catch {
            enc.endEncoding() // Metal aborts on encoders released mid-encode.
            throw error
        }
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return dispatches
    }

    static func floats(_ buf: MTLBuffer, _ n: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: buf.contents().bindMemory(to: Float.self, capacity: n), count: n))
    }

    static func expectBitwise(_ a: MTLBuffer, _ b: MTLBuffer, _ n: Int, _ label: String) {
        #expect(floats(a, n).map(\.bitPattern) == floats(b, n).map(\.bitPattern),
                "\(label): batched kernel diverges bitwise from per-token dispatches")
    }

    /// norm_copy_batch: per-token norm_copy dispatches vs one batched
    /// dispatch over strided hidden rows and destinations, bitwise.
    @Test func normCopyBatchMatchesPerTokenBitwise() throws {
        let engine = try MetalEngine()
        let dim = 64, tokens = 3, hStride = dim + 9, dstStride = dim + 21, dstOff = 5
        var rng = Rand(11)
        let h = (0..<tokens * hStride).map { _ in rng.float() }
        let wgt = (0..<dim).map { _ in rng.float() }
        let hBuf = engine.makeBuffer(h)
        let wBuf = engine.makeBuffer(wgt)
        let dstFloats = dstOff + tokens * dstStride

        struct NormParams {
            var rows: UInt32; var dim: UInt32; var eps: Float
            var hasWeight: UInt32; var scale: Float; var off: UInt32
        }
        let dstA = engine.device.makeBuffer(length: dstFloats * 4)!
        let perToken = try Self.runEncoded(engine) { enc in
            for t in 0..<tokens {
                enc.setComputePipelineState(try engine.pipeline("norm_copy"))
                var p = NormParams(rows: 1, dim: UInt32(dim), eps: 1e-6, hasWeight: 1,
                                   scale: 1, off: UInt32(dstOff + t * dstStride))
                enc.setBuffer(hBuf, offset: t * hStride * 4, index: 0)
                enc.setBuffer(dstA, offset: 0, index: 1)
                enc.setBuffer(wBuf, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: 3)
                engine.dispatchThreads(
                    enc, threads: MTLSize(width: 32, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
        }
        let dstB = engine.device.makeBuffer(length: dstFloats * 4)!
        let batched = try Self.runEncoded(engine) { enc in
            try engine.encodeNormCopyBatch(
                enc, h: hBuf, hByteOffset: 0, hStride: hStride,
                dst: dstB, weight: wBuf, dim: dim, eps: 1e-6,
                dstOff: dstOff, dstStride: dstStride, tokens: tokens)
        }
        #expect(perToken == tokens, "norm_copy: per-token dispatch count")
        #expect(batched == 1, "norm_copy: batched dispatch count")
        Self.expectBitwise(dstA, dstB, dstFloats, "norm_copy")
    }

    /// silu_mul_batch: per-token encodeSiluMul vs one strided batch, bitwise.
    @Test func siluMulBatchMatchesPerTokenBitwise() throws {
        let engine = try MetalEngine()
        let count = 48, tokens = 4, stride = 200
        let gOff = 3, uOff = 61, dstOff = 119
        var rng = Rand(23)
        let src = (0..<tokens * stride).map { _ in rng.float() }

        func run(_ body: (MTLComputeCommandEncoder, MTLBuffer) throws -> Void)
            throws -> (buf: MTLBuffer, dispatches: Int)
        {
            let buf = engine.makeBuffer(src)
            let n = try Self.runEncoded(engine) { try body($0, buf) }
            return (buf, n)
        }
        let perToken = try run { enc, buf in
            for t in 0..<tokens {
                try engine.encodeSiluMul(
                    enc, buf: buf, count: count, gOff: gOff + t * stride,
                    uOff: uOff + t * stride, dstOff: dstOff + t * stride)
            }
        }
        let batched = try run { enc, buf in
            try engine.encodeSiluMulBatch(
                enc, buf: buf, count: count, gOff: gOff, uOff: uOff,
                dstOff: dstOff, stride: stride, tokens: tokens)
        }
        #expect(perToken.dispatches == tokens, "silu_mul: per-token dispatch count")
        #expect(batched.dispatches == 1, "silu_mul: batched dispatch count")
        Self.expectBitwise(perToken.buf, batched.buf, tokens * stride, "silu_mul")
    }

    /// add_inplace_batch: per-token add_inplace dispatches vs one strided
    /// batch over hidden rows, bitwise.
    @Test func addInplaceBatchMatchesPerTokenBitwise() throws {
        let engine = try MetalEngine()
        let count = 64, tokens = 3, hStride = count + 5, rStride = 150, rOff = 7
        var rng = Rand(31)
        let h = (0..<tokens * hStride).map { _ in rng.float() }
        let r = (0..<tokens * rStride).map { _ in rng.float() }
        let rBuf = engine.makeBuffer(r)

        let hA = engine.makeBuffer(h)
        let perToken = try Self.runEncoded(engine) { enc in
            for t in 0..<tokens {
                enc.setComputePipelineState(try engine.pipeline("add_inplace"))
                var p = SIMD2<UInt32>(UInt32(count), UInt32(rOff + t * rStride))
                enc.setBuffer(hA, offset: t * hStride * 4, index: 0)
                enc.setBuffer(rBuf, offset: 0, index: 1)
                enc.setBytes(&p, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 2)
                engine.dispatchThreads(
                    enc, threads: MTLSize(width: count, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(64, count), height: 1, depth: 1))
            }
        }
        let hB = engine.makeBuffer(h)
        let batched = try Self.runEncoded(engine) { enc in
            try engine.encodeAddInplaceBatch(
                enc, h: hB, hStride: hStride, r: rBuf, rOff: rOff, rStride: rStride,
                count: count, tokens: tokens)
        }
        #expect(perToken == tokens, "add_inplace: per-token dispatch count")
        #expect(batched == 1, "add_inplace: batched dispatch count")
        Self.expectBitwise(hA, hB, tokens * hStride, "add_inplace")
    }

    /// weighted_accum_batch: per-token weighted_accum (own weights each) vs
    /// one batched dispatch with a token-major weights table, bitwise.
    @Test func weightedAccumBatchMatchesPerTokenBitwise() throws {
        let engine = try MetalEngine()
        let d = 64, k = 2, tokens = 3, hStride = d + 3, stride = 400
        let dBase = 11, shOff = 11 + k * d + 9, gateOff = shOff + d + 17
        var rng = Rand(47)
        let h = (0..<tokens * hStride).map { _ in rng.float() }
        let data = (0..<tokens * stride).map { _ in rng.float() }
        let weights = (0..<tokens * k).map { _ in rng.float() }
        let dataBuf = engine.makeBuffer(data)

        let hA = engine.makeBuffer(h)
        let perToken = try Self.runEncoded(engine) { enc in
            for t in 0..<tokens {
                enc.setComputePipelineState(try engine.pipeline("weighted_accum"))
                var p = SIMD8<UInt32>(
                    UInt32(d), UInt32(k), UInt32(dBase + t * stride),
                    UInt32(shOff + t * stride), UInt32(gateOff + t * stride), 0, 0, 0)
                enc.setBuffer(hA, offset: t * hStride * 4, index: 0)
                enc.setBuffer(dataBuf, offset: 0, index: 1)
                enc.setBytes(&p, length: MemoryLayout<SIMD8<UInt32>>.stride, index: 2)
                let wk = Array(weights[t * k..<(t + 1) * k])
                wk.withUnsafeBufferPointer {
                    enc.setBytes($0.baseAddress!, length: $0.count * 4, index: 3)
                }
                engine.dispatchThreads(
                    enc, threads: MTLSize(width: d, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(64, d), height: 1, depth: 1))
            }
        }
        let hB = engine.makeBuffer(h)
        let batched = try Self.runEncoded(engine) { enc in
            try engine.encodeWeightedAccumBatch(
                enc, h: hB, hStride: hStride, data: dataBuf,
                count: d, k: k, dBase: dBase, shOff: shOff, gateOff: gateOff,
                stride: stride, weights: weights, tokens: tokens)
        }
        #expect(perToken == tokens, "weighted_accum: per-token dispatch count")
        #expect(batched == 1, "weighted_accum: batched dispatch count")
        Self.expectBitwise(hA, hB, tokens * hStride, "weighted_accum")
    }

    /// deinterleave_qkvz_batch: per-token deinterleaves vs one strided
    /// batch, bitwise across the whole slotted scratch buffer.
    @Test func deinterleaveBatchMatchesPerTokenBitwise() throws {
        let engine = try MetalEngine()
        let nk = 2, dk = 8, rep = 2, dv = 8, tokens = 3
        let keyDim = nk * dk
        let srcOff = 0, baOff = 96, qkvOff = 120, zOff = 190, bOff = 230, aOff = 240
        let stride = 260
        var rng = Rand(53)
        let src = (0..<tokens * stride).map { _ in rng.float() }

        struct DeintParams {
            var nk: UInt32; var dk: UInt32; var rep: UInt32; var dv: UInt32
            var keyDim: UInt32; var srcOff: UInt32; var baOff: UInt32
            var qkvOff: UInt32; var zOff: UInt32; var bOff: UInt32; var aOff: UInt32
        }
        let bufA = engine.makeBuffer(src)
        let perToken = try Self.runEncoded(engine) { enc in
            for t in 0..<tokens {
                enc.setComputePipelineState(try engine.pipeline("deinterleave_qkvz"))
                let b = t * stride
                var p = DeintParams(
                    nk: UInt32(nk), dk: UInt32(dk), rep: UInt32(rep), dv: UInt32(dv),
                    keyDim: UInt32(keyDim), srcOff: UInt32(srcOff + b),
                    baOff: UInt32(baOff + b), qkvOff: UInt32(qkvOff + b),
                    zOff: UInt32(zOff + b), bOff: UInt32(bOff + b), aOff: UInt32(aOff + b))
                enc.setBuffer(bufA, offset: 0, index: 0)
                enc.setBytes(&p, length: MemoryLayout<DeintParams>.stride, index: 1)
                engine.dispatchThreads(
                    enc, threads: MTLSize(width: nk, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(64, nk), height: 1, depth: 1))
            }
        }
        let bufB = engine.makeBuffer(src)
        let batched = try Self.runEncoded(engine) { enc in
            try engine.encodeDeinterleaveBatch(
                enc, data: bufB, nk: nk, dk: dk, rep: rep, dv: dv, keyDim: keyDim,
                srcOff: srcOff, baOff: baOff, qkvOff: qkvOff, zOff: zOff,
                bOff: bOff, aOff: aOff, stride: stride, tokens: tokens)
        }
        #expect(perToken == tokens, "deinterleave: per-token dispatch count")
        #expect(batched == 1, "deinterleave: batched dispatch count")
        Self.expectBitwise(bufA, bufB, tokens * stride, "deinterleave")
    }

    /// attn_q_prep_batch: per-token preps at ascending positions vs one
    /// batched dispatch (position derived from the token grid slot), bitwise.
    @Test func attnQPrepBatchMatchesPerTokenBitwise() throws {
        let engine = try MetalEngine()
        let heads = 4, headDim = 16, rot = 4, tokens = 3, basePosition = 9
        let srcOff = 0
        let dstOff = heads * 2 * headDim + 7
        let stride = dstOff + heads * headDim + 5
        var rng = Rand(61)
        let src = (0..<tokens * stride).map { _ in rng.float() }
        let wgt = (0..<headDim).map { _ in rng.float() }
        let wBuf = engine.makeBuffer(wgt)

        let bufA = engine.makeBuffer(src)
        let perToken = try Self.runEncoded(engine) { enc in
            for t in 0..<tokens {
                try engine.encodeAttnQPrep(
                    enc, data: bufA, weight: wBuf, heads: heads, headDim: headDim,
                    rotaryDims: rot, eps: 1e-6, ropeTheta: 1e7,
                    position: basePosition + t,
                    srcOff: srcOff + t * stride, dstOff: dstOff + t * stride)
            }
        }
        let bufB = engine.makeBuffer(src)
        let batched = try Self.runEncoded(engine) { enc in
            try engine.encodeAttnQPrepBatch(
                enc, data: bufB, weight: wBuf, heads: heads, headDim: headDim,
                rotaryDims: rot, eps: 1e-6, ropeTheta: 1e7,
                basePosition: basePosition, srcOff: srcOff, dstOff: dstOff,
                slotStride: stride, tokens: tokens)
        }
        #expect(perToken == tokens, "attn_q_prep: per-token dispatch count")
        #expect(batched == 1, "attn_q_prep: batched dispatch count")
        Self.expectBitwise(bufA, bufB, tokens * stride, "attn_q_prep")
    }

    /// attn_kv_append_batch: per-token appends at ascending positions vs one
    /// batched dispatch, bitwise over both caches and the scratch.
    @Test func attnKVAppendBatchMatchesPerTokenBitwise() throws {
        let engine = try MetalEngine()
        let kvHeads = 2, headDim = 16, rot = 4, tokens = 3, basePosition = 5
        let kSrcOff = 3
        let vSrcOff = kSrcOff + kvHeads * headDim + 9
        let stride = vSrcOff + kvHeads * headDim + 11
        let cacheFloats = (basePosition + tokens) * kvHeads * headDim
        var rng = Rand(71)
        let src = (0..<tokens * stride).map { _ in rng.float() }
        let wgt = (0..<headDim).map { _ in rng.float() }
        let seed = (0..<cacheFloats).map { _ in rng.float() }
        let wBuf = engine.makeBuffer(wgt)

        func run(_ body: (MTLComputeCommandEncoder, MTLBuffer, MTLBuffer, MTLBuffer) throws -> Void)
            throws -> (data: MTLBuffer, k: MTLBuffer, v: MTLBuffer, dispatches: Int)
        {
            let data = engine.makeBuffer(src)
            let k = engine.makeBuffer(seed)
            let v = engine.makeBuffer(seed)
            let n = try Self.runEncoded(engine) { try body($0, data, k, v) }
            return (data, k, v, n)
        }
        let perToken = try run { enc, data, k, v in
            for t in 0..<tokens {
                try engine.encodeAttnKVAppend(
                    enc, data: data, weight: wBuf, kCache: k, vCache: v,
                    kvHeads: kvHeads, headDim: headDim, rotaryDims: rot,
                    eps: 1e-6, ropeTheta: 1e7, position: basePosition + t,
                    kSrcOff: kSrcOff + t * stride, vSrcOff: vSrcOff + t * stride)
            }
        }
        let batched = try run { enc, data, k, v in
            try engine.encodeAttnKVAppendBatch(
                enc, data: data, weight: wBuf, kCache: k, vCache: v,
                kvHeads: kvHeads, headDim: headDim, rotaryDims: rot,
                eps: 1e-6, ropeTheta: 1e7, basePosition: basePosition,
                kSrcOff: kSrcOff, vSrcOff: vSrcOff, slotStride: stride, tokens: tokens)
        }
        #expect(perToken.dispatches == tokens, "attn_kv_append: per-token dispatch count")
        #expect(batched.dispatches == 1, "attn_kv_append: batched dispatch count")
        Self.expectBitwise(perToken.k, batched.k, cacheFloats, "attn_kv_append k cache")
        Self.expectBitwise(perToken.v, batched.v, cacheFloats, "attn_kv_append v cache")
        Self.expectBitwise(perToken.data, batched.data, tokens * stride, "attn_kv_append data")
    }

    /// attn_decode_gqa_batch: per-token causal attends at ascending kvLens
    /// vs one batched dispatch whose kernel derives token t's kvLen from
    /// basePosition + the token grid slot, bitwise over the whole slotted
    /// scratch. headDim 64 exercises multi-register accumulators.
    @Test func attnDecodeBatchMatchesPerTokenBitwise() throws {
        let engine = try MetalEngine()
        let heads = 4, kvHeads = 2, headDim = 64, tokens = 3, basePosition = 5
        let qoutOff = 3
        let qOff = qoutOff + heads * 2 * headDim + 7
        let outOff = qOff + heads * headDim + 5
        let stride = outOff + heads * headDim + 9
        let cacheFloats = (basePosition + tokens) * kvHeads * headDim
        var rng = Rand(83)
        let src = (0..<tokens * stride).map { _ in rng.float() }
        let kBuf = engine.makeBuffer((0..<cacheFloats).map { _ in rng.float() })
        let vBuf = engine.makeBuffer((0..<cacheFloats).map { _ in rng.float() })

        let bufA = engine.makeBuffer(src)
        let perToken = try Self.runEncoded(engine) { enc in
            for t in 0..<tokens {
                try engine.encodeAttnDecode(
                    enc, data: bufA, kCache: kBuf, vCache: vBuf,
                    heads: heads, kvHeads: kvHeads, headDim: headDim,
                    kvLen: basePosition + t + 1,
                    qOff: qOff + t * stride, qoutOff: qoutOff + t * stride,
                    outOff: outOff + t * stride)
            }
        }
        let bufB = engine.makeBuffer(src)
        let batched = try Self.runEncoded(engine) { enc in
            try engine.encodeAttnDecodeBatch(
                enc, data: bufB, kCache: kBuf, vCache: vBuf,
                heads: heads, kvHeads: kvHeads, headDim: headDim,
                basePosition: basePosition, qOff: qOff, qoutOff: qoutOff,
                outOff: outOff, slotStride: stride, tokens: tokens)
        }
        #expect(perToken == tokens, "attn_decode: per-token dispatch count")
        #expect(batched == 1, "attn_decode: batched dispatch count")
        Self.expectBitwise(bufA, bufB, tokens * stride, "attn_decode")
    }

    /// delta_pre_batch: per-token delta_pre dispatches vs one strided batch
    /// over the slots' a/b rows, bitwise (g and beta both land at outOff).
    @Test func deltaPreBatchMatchesPerTokenBitwise() throws {
        let engine = try MetalEngine()
        let nv = 6, tokens = 3
        let aOff = 5, bOff = aOff + nv + 3, outOff = bOff + nv + 7
        let stride = outOff + 2 * nv + 9
        var rng = Rand(113)
        let src = (0..<tokens * stride).map { _ in rng.float() }
        let aLog = engine.makeBuffer((0..<nv).map { _ in rng.float() })
        let dtBias = engine.makeBuffer((0..<nv).map { _ in rng.float() })

        let bufA = engine.makeBuffer(src)
        let perToken = try Self.runEncoded(engine) { enc in
            for t in 0..<tokens {
                enc.setComputePipelineState(try engine.pipeline("delta_pre"))
                var p = SIMD4<UInt32>(
                    UInt32(nv), UInt32(aOff + t * stride),
                    UInt32(bOff + t * stride), UInt32(outOff + t * stride))
                enc.setBuffer(bufA, offset: 0, index: 0)
                enc.setBuffer(aLog, offset: 0, index: 1)
                enc.setBuffer(dtBias, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 3)
                engine.dispatchThreads(
                    enc, threads: MTLSize(width: nv, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(64, nv), height: 1, depth: 1))
            }
        }
        let bufB = engine.makeBuffer(src)
        let batched = try Self.runEncoded(engine) { enc in
            try engine.encodeDeltaPreBatch(
                enc, data: bufB, aLog: aLog, dtBias: dtBias,
                nv: nv, aOff: aOff, bOff: bOff, outOff: outOff,
                slotStride: stride, tokens: tokens)
        }
        #expect(perToken == tokens, "delta_pre: per-token dispatch count")
        #expect(batched == 1, "delta_pre: batched dispatch count")
        Self.expectBitwise(bufA, bufB, tokens * stride, "delta_pre")
    }

    /// rmsnorm_rows_batch: per-token in-place rmsnorm_rows dispatches vs one
    /// strided batch, bitwise — in the weightless scaled shape the DeltaNet
    /// q/k norms use.
    @Test func rmsnormRowsBatchMatchesPerTokenBitwise() throws {
        let engine = try MetalEngine()
        let rows = 2, dim = 32, tokens = 3, off = 5
        let stride = rows * dim + 11
        let scale: Float = 0.25
        let total = off + (tokens - 1) * stride + rows * dim
        var rng = Rand(127)
        let src = (0..<total).map { _ in rng.float() }

        struct NormParams {
            var rows: UInt32; var dim: UInt32; var eps: Float
            var hasWeight: UInt32; var scale: Float; var off: UInt32
        }
        let bufA = engine.makeBuffer(src)
        let perToken = try Self.runEncoded(engine) { enc in
            for t in 0..<tokens {
                enc.setComputePipelineState(try engine.pipeline("rmsnorm_rows"))
                var p = NormParams(rows: UInt32(rows), dim: UInt32(dim), eps: 1e-6,
                                   hasWeight: 0, scale: scale, off: UInt32(off + t * stride))
                enc.setBuffer(bufA, offset: 0, index: 0)
                enc.setBuffer(bufA, offset: 0, index: 1)
                enc.setBytes(&p, length: MemoryLayout<NormParams>.stride, index: 2)
                engine.dispatchThreads(
                    enc, threads: MTLSize(width: 32, height: rows, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
        }
        let bufB = engine.makeBuffer(src)
        let batched = try Self.runEncoded(engine) { enc in
            try engine.encodeRMSNormRowsBatch(
                enc, data: bufB, weight: nil, off: off, rows: rows, dim: dim,
                scale: scale, eps: 1e-6, slotStride: stride, tokens: tokens)
        }
        #expect(perToken == tokens, "rmsnorm_rows: per-token dispatch count")
        #expect(batched == 1, "rmsnorm_rows: batched dispatch count")
        Self.expectBitwise(bufA, bufB, total, "rmsnorm_rows")
    }

    /// gated_norm_mul_batch: per-token gated_norm_mul dispatches vs one
    /// batched dispatch with per-stream strides — y rows in a contiguous
    /// staging block (the chunked recurrence's layout), z and the output
    /// slot-strided — bitwise.
    @Test func gatedNormMulBatchMatchesPerTokenBitwise() throws {
        let engine = try MetalEngine()
        let nv = 4, dv = 8, tokens = 3
        let zOff = 3
        let outOff = zOff + nv * dv + 5
        let slotStride = outOff + nv * dv + 7
        let yOff = tokens * slotStride
        let yStride = nv * dv
        let total = yOff + tokens * yStride
        var rng = Rand(137)
        let src = (0..<total).map { _ in rng.float() }
        let wgt = engine.makeBuffer((0..<dv).map { _ in rng.float() })

        struct GatedNormParams {
            var nv: UInt32; var dv: UInt32; var eps: Float
            var yOff: UInt32; var zOff: UInt32; var outOff: UInt32
        }
        let bufA = engine.makeBuffer(src)
        let perToken = try Self.runEncoded(engine) { enc in
            for t in 0..<tokens {
                enc.setComputePipelineState(try engine.pipeline("gated_norm_mul"))
                var p = GatedNormParams(
                    nv: UInt32(nv), dv: UInt32(dv), eps: 1e-6,
                    yOff: UInt32(yOff + t * yStride), zOff: UInt32(zOff + t * slotStride),
                    outOff: UInt32(outOff + t * slotStride))
                enc.setBuffer(bufA, offset: 0, index: 0)
                enc.setBuffer(wgt, offset: 0, index: 1)
                enc.setBytes(&p, length: MemoryLayout<GatedNormParams>.stride, index: 2)
                engine.dispatchThreads(
                    enc, threads: MTLSize(width: 32, height: nv, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
        }
        let bufB = engine.makeBuffer(src)
        let batched = try Self.runEncoded(engine) { enc in
            try engine.encodeGatedNormMulBatch(
                enc, data: bufB, weight: wgt, nv: nv, dv: dv, eps: 1e-6,
                yOff: yOff, yStride: yStride, zOff: zOff, zStride: slotStride,
                outOff: outOff, outStride: slotStride, tokens: tokens)
        }
        #expect(perToken == tokens, "gated_norm_mul: per-token dispatch count")
        #expect(batched == 1, "gated_norm_mul: batched dispatch count")
        Self.expectBitwise(bufA, bufB, total, "gated_norm_mul")
    }

    // MARK: - S1b chunked DeltaNet recurrence (T-step scan)

    /// gated_delta_step's T parameter is a sequential in-dispatch scan: one
    /// T=5 dispatch must reproduce five chained T=1 dispatches bitwise (all
    /// y rows and the final state), because the step loop performs the
    /// identical ascending arithmetic with state carried in f32 registers
    /// instead of a per-token f32 device round trip. This is the oracle the
    /// chunked prefill recurrence stands on.
    @Test func gatedDeltaChunkedScanMatchesSequentialBitwise() throws {
        let engine = try MetalEngine()
        let T = 5, Hk = 2, Hv = 4, Dk = 32, Dv = 8
        var rng = Rand(97)
        let q = (0..<T * Hk * Dk).map { _ in rng.float() }
        let k = (0..<T * Hk * Dk).map { _ in rng.float() }
        let v = (0..<T * Hv * Dv).map { _ in rng.float() }
        let g = (0..<T * Hv).map { _ in abs(rng.float()) * 0.9 }
        let beta = (0..<T * Hv).map { _ in abs(rng.float()) }
        let state0 = (0..<Hv * Dv * Dk).map { _ in rng.float() * 0.1 }

        var seqY: [Float] = []
        var seqState = state0
        for t in 0..<T {
            let (y, s) = try engine.gatedDeltaStep(
                q: Array(q[t * Hk * Dk..<(t + 1) * Hk * Dk]),
                k: Array(k[t * Hk * Dk..<(t + 1) * Hk * Dk]),
                v: Array(v[t * Hv * Dv..<(t + 1) * Hv * Dv]),
                g: Array(g[t * Hv..<(t + 1) * Hv]),
                beta: Array(beta[t * Hv..<(t + 1) * Hv]),
                state: seqState, T: 1, Hk: Hk, Hv: Hv, Dk: Dk, Dv: Dv
            )
            seqY.append(contentsOf: y)
            seqState = s
        }

        let (y, state) = try engine.gatedDeltaStep(
            q: q, k: k, v: v, g: g, beta: beta, state: state0,
            T: T, Hk: Hk, Hv: Hv, Dk: Dk, Dv: Dv
        )
        #expect(y.map(\.bitPattern) == seqY.map(\.bitPattern),
                "T-step scan y diverges bitwise from chained T=1 steps")
        #expect(state.map(\.bitPattern) == seqState.map(\.bitPattern),
                "T-step scan state diverges bitwise from chained T=1 steps")
    }

    /// conv_scan is the same kind of sequential in-dispatch scan for the
    /// depthwise conv: one T=3 dispatch must reproduce three chained
    /// conv_step dispatches bitwise (every token's conv+silu row and the
    /// final shifted history), because each iteration performs the
    /// single-step kernel's arithmetic verbatim with the history carried
    /// across tokens in registers instead of a barrier-fenced device round
    /// trip per token.
    @Test func convScanMatchesChainedStepsBitwise() throws {
        let engine = try MetalEngine()
        let convDim = 48, K = 4, tokens = 3
        let inOff = 5
        let outOff = inOff + convDim + 3
        let stride = outOff + convDim + 7
        var rng = Rand(151)
        let src = (0..<tokens * stride).map { _ in rng.float() }
        let hist0 = (0..<(K - 1) * convDim).map { _ in rng.float() }
        let wgt = engine.makeBuffer((0..<convDim * K).map { _ in rng.float() })

        let dataA = engine.makeBuffer(src)
        let histA = engine.makeBuffer(hist0)
        let perToken = try Self.runEncoded(engine) { enc in
            for t in 0..<tokens {
                enc.setComputePipelineState(try engine.pipeline("conv_step"))
                var p = SIMD4<UInt32>(UInt32(convDim), UInt32(K),
                                      UInt32(inOff + t * stride), UInt32(outOff + t * stride))
                enc.setBuffer(dataA, offset: 0, index: 0)
                enc.setBuffer(histA, offset: 0, index: 1)
                enc.setBuffer(wgt, offset: 0, index: 2)
                enc.setBytes(&p, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 3)
                engine.dispatchThreads(
                    enc, threads: MTLSize(width: convDim, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: min(64, convDim), height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers) // token t+1 reads the advanced history
            }
        }
        let dataB = engine.makeBuffer(src)
        let histB = engine.makeBuffer(hist0)
        let scanned = try Self.runEncoded(engine) { enc in
            try engine.encodeConvScan(
                enc, data: dataB, hist: histB, weight: wgt, convDim: convDim, K: K,
                inOff: inOff, outOff: outOff, slotStride: stride, tokens: tokens)
        }
        #expect(perToken == tokens, "conv_step: per-token dispatch count")
        #expect(scanned == 1, "conv_scan: scan dispatch count")
        Self.expectBitwise(dataA, dataB, tokens * stride, "conv_scan data")
        Self.expectBitwise(histA, histB, (K - 1) * convDim, "conv_scan history")
    }

    /// delta_gather packs slot-strided q/k/v/g/beta into the contiguous
    /// [T, row] staging blocks as pure copies — bitwise against a CPU
    /// reference, one dispatch, sources untouched.
    @Test func deltaGatherPacksSlotsBitwise() throws {
        let engine = try MetalEngine()
        let keyDim = 16, valueDim = 32, nv = 4, tokens = 3
        let convOff = 7
        let gbOff = convOff + 2 * keyDim + valueDim + 5
        let slotStride = gbOff + 2 * nv + 9
        let stageBase = tokens * slotStride
        let qOut = stageBase
        let kOut = qOut + tokens * keyDim
        let vOut = kOut + tokens * keyDim
        let gOut = vOut + tokens * valueDim
        let bOut = gOut + tokens * nv
        let total = bOut + tokens * nv
        var rng = Rand(101)
        let data = (0..<total).map { _ in rng.float() }

        var expected = data
        for t in 0..<tokens {
            let slot = t * slotStride
            for i in 0..<keyDim {
                expected[qOut + t * keyDim + i] = data[slot + convOff + i]
                expected[kOut + t * keyDim + i] = data[slot + convOff + keyDim + i]
            }
            for i in 0..<valueDim {
                expected[vOut + t * valueDim + i] = data[slot + convOff + 2 * keyDim + i]
            }
            for i in 0..<nv {
                expected[gOut + t * nv + i] = data[slot + gbOff + i]
                expected[bOut + t * nv + i] = data[slot + gbOff + nv + i]
            }
        }

        let buf = engine.makeBuffer(data)
        let dispatches = try Self.runEncoded(engine) { enc in
            try engine.encodeDeltaGather(
                enc, data: buf, keyDim: keyDim, valueDim: valueDim, nv: nv,
                slotStride: slotStride, convOff: convOff, gbOff: gbOff,
                qOut: qOut, kOut: kOut, vOut: vOut, gOut: gOut, bOut: bOut,
                tokens: tokens)
        }
        #expect(dispatches == 1, "delta_gather: dispatch count")
        #expect(Self.floats(buf, total).map(\.bitPattern) == expected.map(\.bitPattern),
                "delta_gather diverges from the CPU copy reference")
    }
}
