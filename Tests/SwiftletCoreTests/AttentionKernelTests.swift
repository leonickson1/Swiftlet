import Foundation
import Metal
import Testing
@testable import SwiftletCore

/// S2 attention kernels vs the exact CPU math they replace (the fast path's
/// CPU attention core): query extraction + RMSNorm + partial RoPE, KV append
/// into position-indexed cache rows (K normed + roped, V verbatim), and gated
/// GQA softmax attention over the cached rows.
@Suite struct AttentionKernelTests {
    struct Geometry {
        let H: Int      // query heads
        let KVH: Int    // kv heads
        let hd: Int     // head dim
        let rot: Int    // rotary dims
    }

    /// Test-local copy of the reference partial-RoPE (QwenCPUModel.applyRope
    /// math with an explicit position per call).
    static func rope(
        _ x: inout [Float], heads: Int, hd: Int, rot: Int, theta: Float, position: Int
    ) {
        let half = rot / 2
        for head in 0..<heads {
            let base = head * hd
            for j in 0..<half {
                let invFreq = powf(theta, -Float(2 * j) / Float(rot))
                let angle = Float(position) * invFreq
                let c = cosf(angle), sn = sinf(angle)
                let a = x[base + j]
                let b = x[base + half + j]
                x[base + j] = a * c - b * sn
                x[base + half + j] = b * c + a * sn
            }
        }
    }

    static func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        precondition(a.count == b.count)
        var m: Float = 0
        for i in a.indices { m = max(m, abs(a[i] - b[i])) }
        return m
    }

    /// Runs the three kernels in one command buffer against a scratch layout
    /// shaped like the fast path's regions, then compares every artifact —
    /// prepped q, the appended cache rows, and the gated context — to the CPU
    /// reference. `past` cache rows already hold earlier positions, so the
    /// softmax really spans history, not just the new token.
    static func compare(_ geo: Geometry, seed: UInt64, past: Int = 3) throws {
        let engine = try MetalEngine()
        let H = geo.H, KVH = geo.KVH, hd = geo.hd, rot = geo.rot
        let position = past
        let theta: Float = 10_000_000
        let eps: Float = 1e-6
        var rng = MetalKernelTests.Rand(seed)

        // Raw projection outputs for the new token, plus norm weights and a
        // cache history of already-processed rows.
        let qout = (0..<H * 2 * hd).map { _ in rng.float() }
        let knew = (0..<KVH * hd).map { _ in rng.float() }
        let vnew = (0..<KVH * hd).map { _ in rng.float() }
        let qNorm = (0..<hd).map { _ in 1 + 0.25 * rng.float() }
        let kNorm = (0..<hd).map { _ in 1 + 0.25 * rng.float() }
        let kPast = (0..<past * KVH * hd).map { _ in rng.float() }
        let vPast = (0..<past * KVH * hd).map { _ in rng.float() }

        // CPU reference: the former attnCoreCPU math, op for op.
        var qRef = [Float](repeating: 0, count: H * hd)
        var gate = [Float](repeating: 0, count: H * hd)
        for head in 0..<H {
            for i in 0..<hd {
                qRef[head * hd + i] = qout[head * 2 * hd + i]
                gate[head * hd + i] = qout[head * 2 * hd + hd + i]
            }
        }
        QwenCPUModel.rmsNorm(&qRef, rows: H, dim: hd, weight: qNorm, eps: eps)
        var kRef = knew
        QwenCPUModel.rmsNorm(&kRef, rows: KVH, dim: hd, weight: kNorm, eps: eps)
        Self.rope(&qRef, heads: H, hd: hd, rot: rot, theta: theta, position: position)
        Self.rope(&kRef, heads: KVH, hd: hd, rot: rot, theta: theta, position: position)
        let kAll = kPast + kRef
        let vAll = vPast + vnew
        let kvLen = past + 1
        let scale = 1 / Float(hd).squareRoot()
        let group = H / KVH
        var outRef = [Float](repeating: 0, count: H * hd)
        var scores = [Float](repeating: 0, count: kvLen)
        for head in 0..<H {
            let kvHead = head / group
            for sj in 0..<kvLen {
                var dot: Float = 0
                for i in 0..<hd { dot += qRef[head * hd + i] * kAll[(sj * KVH + kvHead) * hd + i] }
                scores[sj] = dot * scale
            }
            QwenCPUModel.softmaxRow(&scores, base: 0, count: kvLen)
            for sj in 0..<kvLen {
                let p = scores[sj]
                let vBase = (sj * KVH + kvHead) * hd
                for i in 0..<hd { outRef[head * hd + i] += p * vAll[vBase + i] }
            }
        }
        for i in 0..<outRef.count { outRef[i] *= QwenCPUModel.sigmoid(gate[i]) }

        // GPU: fast-path-shaped scratch regions in one shared buffer.
        let qoutOff = 0
        let kOff = qoutOff + H * 2 * hd
        let vOff = kOff + KVH * hd
        let qprepOff = vOff + KVH * hd
        let outOff = qprepOff + H * hd
        let total = outOff + H * hd
        var scratchInit = [Float](repeating: 0, count: total)
        scratchInit.replaceSubrange(qoutOff..<qoutOff + qout.count, with: qout)
        scratchInit.replaceSubrange(kOff..<kOff + knew.count, with: knew)
        scratchInit.replaceSubrange(vOff..<vOff + vnew.count, with: vnew)
        let scratch = engine.makeBuffer(scratchInit)
        let qNormBuf = engine.makeBuffer(qNorm)
        let kNormBuf = engine.makeBuffer(kNorm)
        let kCache = engine.makeBuffer(kPast + [Float](repeating: 0, count: KVH * hd))
        let vCache = engine.makeBuffer(vPast + [Float](repeating: 0, count: KVH * hd))

        let cb = engine.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        try engine.encodeAttnQPrep(
            enc, data: scratch, weight: qNormBuf,
            heads: H, headDim: hd, rotaryDims: rot, eps: eps, ropeTheta: theta,
            position: position, srcOff: qoutOff, dstOff: qprepOff
        )
        try engine.encodeAttnKVAppend(
            enc, data: scratch, weight: kNormBuf, kCache: kCache, vCache: vCache,
            kvHeads: KVH, headDim: hd, rotaryDims: rot, eps: eps, ropeTheta: theta,
            position: position, kSrcOff: kOff, vSrcOff: vOff
        )
        enc.memoryBarrier(scope: .buffers)
        try engine.encodeAttnDecode(
            enc, data: scratch, kCache: kCache, vCache: vCache,
            heads: H, kvHeads: KVH, headDim: hd, kvLen: kvLen,
            qOff: qprepOff, qoutOff: qoutOff, outOff: outOff
        )
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.status == .completed, "attention kernel command buffer failed")

        func read(_ off: Int, _ n: Int) -> [Float] {
            Array(UnsafeBufferPointer(
                start: scratch.contents().advanced(by: off * 4)
                    .bindMemory(to: Float.self, capacity: n),
                count: n))
        }
        func readCache(_ buf: MTLBuffer, _ row: Int, _ n: Int) -> [Float] {
            Array(UnsafeBufferPointer(
                start: buf.contents().advanced(by: row * n * 4)
                    .bindMemory(to: Float.self, capacity: n),
                count: n))
        }

        let label = "H=\(H) KVH=\(KVH) hd=\(hd) rot=\(rot)"
        let qDiff = Self.maxAbsDiff(read(qprepOff, H * hd), qRef)
        #expect(qDiff < 1e-5, "\(label): prepped q maxAbsDiff \(qDiff)")
        let kDiff = Self.maxAbsDiff(readCache(kCache, past, KVH * hd), kRef)
        #expect(kDiff < 1e-5, "\(label): appended K row maxAbsDiff \(kDiff)")
        let vDiff = Self.maxAbsDiff(readCache(vCache, past, KVH * hd), vnew)
        #expect(vDiff == 0, "\(label): appended V row must copy verbatim, diff \(vDiff)")
        let pastKDiff = Self.maxAbsDiff(readCache(kCache, 0, past * KVH * hd), kPast)
        #expect(pastKDiff == 0, "\(label): append disturbed earlier K rows")
        let outDiff = Self.maxAbsDiff(read(outOff, H * hd), outRef)
        #expect(outDiff < 1e-5, "\(label): gated attention output maxAbsDiff \(outDiff)")
    }

    /// Tiny-fixture geometry: exercises head dims below one SIMD width.
    @Test func attentionKernelsMatchCPUCoreTinyHeads() throws {
        try Self.compare(Geometry(H: 4, KVH: 2, hd: 16, rot: 4), seed: 11)
    }

    /// 80B-class geometry: multi-register accumulators (hd > 32) and a
    /// deeper GQA group.
    @Test func attentionKernelsMatchCPUCoreWideHeads() throws {
        try Self.compare(Geometry(H: 8, KVH: 2, hd: 128, rot: 32), seed: 23)
    }

    /// A KV history long enough that every cooperating simdgroup in the
    /// attend owns many cache rows, with a row count off every power-of-two
    /// grid so the tail is uneven — the softmax must span all of it.
    @Test func attentionKernelsMatchCPUCoreLongHistory() throws {
        try Self.compare(Geometry(H: 4, KVH: 2, hd: 128, rot: 32), seed: 31, past: 133)
    }

    /// A head dim off the float4 grid (18 % 4 != 0) keeps the lane-strided
    /// scalar reads honest for any vectorized-load fast path.
    @Test func attentionKernelsMatchCPUCoreOddHeadDim() throws {
        try Self.compare(Geometry(H: 4, KVH: 2, hd: 18, rot: 6), seed: 37, past: 9)
    }
}
