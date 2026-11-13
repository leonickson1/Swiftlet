import Foundation
import Metal
import Testing
@testable import SwiftletCore

/// The expert cache's placement policy is part of the measured behaviour
/// (hit/miss counts in every bench log), so the S3c fast path -- flat-array
/// victim scan, batched concurrent preads -- must reproduce the original
/// dictionary-scanned, one-pread-at-a-time policy decision for decision.
@Suite struct ExpertCacheTests {
    static let fixturesDir = MetalModelTests.fixturesDir

    /// Verbatim transcription of the pre-S3c policy (`buffers` +
    /// `slotForFill`, slot allocation simulated): LFU with history, recency
    /// tie-break, lowest slot on full ties, batch members protected.
    struct ReferencePolicy {
        let maxSlots: Int
        /// LRU instead of LFU: every slot ranks at frequency 0.
        var lru = false
        var slotKey: [Int64] = []
        var keyToSlot: [Int64: Int] = [:]
        var freq: [Int64: Int] = [:]
        var lastUse: [Int: Int] = [:]
        var tick = 0
        var hits = 0
        var misses = 0

        func key(_ layer: Int, _ expert: Int) -> Int64 { Int64(layer) << 32 | Int64(expert) }

        mutating func buffers(layer: Int, experts: [Int]) -> [Int] {
            tick += 1
            var protectedSlots = Set<Int>()
            var result: [Int] = []
            for e in experts {
                let k = key(layer, e)
                freq[k, default: 0] += 1
                if let s = keyToSlot[k] {
                    hits += 1
                    lastUse[s] = tick
                    protectedSlots.insert(s)
                    result.append(s)
                    continue
                }
                misses += 1
                let s = slotForFill(excluding: protectedSlots)
                if slotKey[s] >= 0 { keyToSlot.removeValue(forKey: slotKey[s]) }
                slotKey[s] = k
                keyToSlot[k] = s
                lastUse[s] = tick
                protectedSlots.insert(s)
                result.append(s)
            }
            return result
        }

        mutating func slotForFill(excluding: Set<Int>) -> Int {
            if slotKey.count < maxSlots {
                slotKey.append(-1)
                return slotKey.count - 1
            }
            if let free = slotKey.firstIndex(of: -1) { return free }
            var victim = -1
            var victimScore = (Int.max, Int.max)
            for s in 0..<slotKey.count where !excluding.contains(s) {
                let f = lru ? 0 : freq[slotKey[s]] ?? 0
                let score = (f, lastUse[s] ?? 0)
                if score < victimScore {
                    victimScore = score
                    victim = s
                }
            }
            precondition(victim >= 0)
            return victim
        }
    }

    static func repackedTiny() throws -> URL {
        let src = fixturesDir.appendingPathComponent("tiny-model-q4")
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-q4-cache-\(UUID().uuidString).qpack")
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        try repacker.repack()
        return out
    }

    /// The cache now takes an explicit physical budget and refuses one too
    /// small for min(16, total) slots. Size a budget that pins the pool to
    /// exactly `slots` blobs (the repacker page-aligns the stride, so the
    /// per-slot allocation equals the stride).
    static func budget(slots: Int, container: URL) throws -> Int {
        let reader = try QpackExpertReader(containerDir: container)
        return slots * reader.layout.expertStride
    }

    /// A seeded pseudo-random request stream (skewed toward a hot subset,
    /// batches of 1-6 distinct experts) over 16 slots of the 64-blob tiny
    /// container: every hit/miss count and every slot placement must match
    /// the reference policy, request by request.
    @Test(arguments: [ExpertCache.EvictionPolicy.lfu, .lru])
    func placementMatchesTheReferencePolicy(policy: ExpertCache.EvictionPolicy) throws {
        let out = try Self.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = try ExpertCache(
            containerDir: out, device: device,
            budgetBytes: Self.budget(slots: 16, container: out))
        #expect(cache.slotCount == 16, "a 16-slot budget must yield 16 slots")
        cache.policy = policy
        var reference = ReferencePolicy(maxSlots: cache.slotCount, lru: policy == .lru)
        let layers = cache.reader.layout.layerCount
        let expertsPerLayer = cache.reader.layout.expertCount

        var rng = UInt64(0x9E3779B97F4A7C15)
        func next(_ n: Int) -> Int {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return Int((rng >> 33) % UInt64(n))
        }
        for request in 0..<400 {
            let layer = next(layers)
            let batch = 1 + next(6)
            var experts: [Int] = []
            while experts.count < batch {
                // 70% of picks from a hot half of the layer's experts.
                let e = next(10) < 7 ? next(expertsPerLayer / 2) : next(expertsPerLayer)
                if !experts.contains(e) { experts.append(e) }
            }
            let expected = reference.buffers(layer: layer, experts: experts)
            let got = try cache.buffers(layer: layer, experts: experts)
            #expect(got.count == experts.count, "request \(request): result count")
            for (i, e) in experts.enumerated() {
                #expect(cache.residentSlot(layer: layer, expert: e) == expected[i],
                        "request \(request): (\(layer), \(e)) placed in a different slot")
            }
            #expect(cache.hits == reference.hits && cache.misses == reference.misses,
                    "request \(request): hit/miss counts diverged (\(cache.hits)/\(cache.misses) vs \(reference.hits)/\(reference.misses))")
            if cache.hits != reference.hits || cache.misses != reference.misses { break }
        }
        #expect(cache.misses > cache.slotCount, "stream never evicted; the policy was not exercised")
        #expect(cache.hits > 0)
    }

    /// Batches with several misses go through the concurrent path: every
    /// returned buffer must hold exactly the blob an independent single
    /// pread returns, through growth, eviction, and refill.
    @Test func concurrentFillsLandTheRightBytes() throws {
        let out = try Self.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = try ExpertCache(
            containerDir: out, device: device,
            budgetBytes: Self.budget(slots: 16, container: out))
        let oracle = try QpackExpertReader(containerDir: out)
        let stride = cache.stride
        var scratch = [UInt8](repeating: 0, count: stride)
        let all = Array(0..<cache.reader.layout.expertCount)

        func check(layer: Int, experts: [Int], label: String) throws {
            let bufs = try cache.buffers(layer: layer, experts: experts)
            for (i, e) in experts.enumerated() {
                try scratch.withUnsafeMutableBytes {
                    try oracle.readExpert(layer: layer, expert: e, into: $0.baseAddress!)
                }
                let got = Array(UnsafeRawBufferPointer(start: bufs[i].contents(), count: stride))
                #expect(got == scratch, "\(label): layer \(layer) expert \(e) bytes diverged")
                #expect(cache.residentSlot(layer: layer, expert: e) != nil, "\(label): not resident")
            }
        }
        try check(layer: 3, experts: all, label: "growth")          // 8 misses, slots 0-7
        try check(layer: 5, experts: all, label: "growth 2")        // 8 misses, slots 8-15
        try check(layer: 1, experts: all.reversed(), label: "evict") // 8 misses, evictions
        try check(layer: 3, experts: all, label: "refill")           // misses again
        try check(layer: 3, experts: [2, 4], label: "hits")
        #expect(cache.allocatedSlots == 16)
    }

    /// A short read must throw, and the cache must not advertise the batch's
    /// keys as resident afterwards.
    @Test func failedReadLeavesNoResidentKey() throws {
        let out = try Self.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = try ExpertCache(
            containerDir: out, device: device,
            budgetBytes: Self.budget(slots: 16, container: out))
        // Truncate layer 2's file to two and a half blobs.
        let file = out.appendingPathComponent("packed_experts/layer_02.bin")
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: UInt64(cache.stride * 5 / 2))
        try handle.close()

        _ = try cache.buffers(layer: 2, experts: [0, 1])
        #expect(throws: (any Error).self) { try cache.buffers(layer: 2, experts: [2, 3, 4]) }
        for e in [2, 3, 4] {
            #expect(cache.residentSlot(layer: 2, expert: e) == nil, "expert \(e) resident after a short read")
        }
        #expect(cache.residentSlot(layer: 2, expert: 0) != nil, "earlier fills must survive")
        #expect(cache.allocatedSlots == 5, "the batch's slots stay allocated, just free")
        // The cache keeps working; the freed slots are the first victims once
        // the pool is full (free slots precede eviction in slotForFill).
        let after = try cache.buffers(layer: 4, experts: [0, 1, 2])
        #expect(after.count == 3)
        #expect(cache.residentSlot(layer: 4, expert: 2) != nil)
    }
}
