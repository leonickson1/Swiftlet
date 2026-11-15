import Foundation
import Metal
import Testing
@testable import SwiftletCore

/// Router lookahead and background expert prefetch may change when bytes
/// arrive and what stays cached, never what the model computes.
@Suite struct LookaheadPrefetchTests {
    /// A prefetched blob is byte-identical to the file, and requesting it
    /// counts as a hit on a used prefetch.
    @Test func prefetchedBlobsMatchTheFile() throws {
        let out = try ExpertCacheTests.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = try ExpertCache(
            containerDir: out, device: device,
            budgetBytes: ExpertCacheTests.budget(slots: 16, container: out))
        let oracle = try QpackExpertReader(containerDir: out)
        var scratch = [UInt8](repeating: 0, count: cache.stride)

        try cache.prefetch(layer: 3, experts: [0, 2, 5, 7])
        let bufs = try cache.buffers(layer: 3, experts: [0, 2, 5, 7])
        for (i, e) in [0, 2, 5, 7].enumerated() {
            try scratch.withUnsafeMutableBytes {
                try oracle.readExpert(layer: 3, expert: e, into: $0.baseAddress!)
            }
            let got = Array(UnsafeRawBufferPointer(start: bufs[i].contents(), count: cache.stride))
            #expect(got == scratch, "layer 3 expert \(e) bytes diverged after prefetch")
        }
        #expect(cache.prefetchIssued == 4)
        #expect(cache.prefetchUsed == 4)
        #expect(cache.hits == 4 && cache.misses == 0)

        // Already resident: nothing to read ahead.
        try cache.prefetch(layer: 3, experts: [0, 2])
        #expect(cache.prefetchIssued == 4)
    }

    /// The slots the latest `buffers` call handed out feed the next command
    /// buffer, which runs while a prefetch reads; the prefetch must evict
    /// around them, and their bytes must stay intact.
    @Test func prefetchNeverEvictsTheLastBatch() throws {
        let out = try ExpertCacheTests.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = try ExpertCache(
            containerDir: out, device: device,
            budgetBytes: ExpertCacheTests.budget(slots: 16, container: out))
        let oracle = try QpackExpertReader(containerDir: out)
        let all = Array(0..<cache.reader.layout.expertCount)
        var scratch = [UInt8](repeating: 0, count: cache.stride)

        cache.policy = .lru
        _ = try cache.buffers(layer: 2, experts: [0])                   // slot 0
        _ = try cache.buffers(layer: 1, experts: all)                   // slots 1-8
        _ = try cache.buffers(layer: 3, experts: Array(all.prefix(7)))  // slots 9-15
        let gpu = try cache.buffers(layer: 2, experts: [0])             // the batch the GPU now reads
        #expect(cache.residentSlot(layer: 2, expert: 0) == 0)
        // Layer 1 then layer 3 go first; after that every slot ties on
        // recency with slot 0, which the lowest-slot tie-break would pick.
        try cache.prefetch(layer: 4, experts: all)
        cache.drainPrefetches()
        try cache.prefetch(layer: 5, experts: all)
        cache.drainPrefetches()

        #expect(cache.residentSlot(layer: 2, expert: 0) == 0, "the last batch's slot was evicted")
        try scratch.withUnsafeMutableBytes {
            try oracle.readExpert(layer: 2, expert: 0, into: $0.baseAddress!)
        }
        let got = Array(UnsafeRawBufferPointer(start: gpu[0].contents(), count: cache.stride))
        #expect(got == scratch, "the last batch's bytes were overwritten")
        #expect(cache.prefetchIssued == 16)
    }

    /// A background read that fails must not leave its experts looking
    /// resident; a later request reads them on demand (and here fails too).
    @Test func failedPrefetchFreesItsSlots() throws {
        let out = try ExpertCacheTests.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = try ExpertCache(
            containerDir: out, device: device,
            budgetBytes: ExpertCacheTests.budget(slots: 16, container: out))
        let file = out.appendingPathComponent("packed_experts/layer_02.bin")
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: UInt64(cache.stride * 5 / 2))
        try handle.close()

        try cache.prefetch(layer: 2, experts: [3, 4])
        cache.drainPrefetches()
        #expect(cache.residentSlot(layer: 2, expert: 3) == nil)
        #expect(cache.residentSlot(layer: 2, expert: 4) == nil)
        #expect(throws: (any Error).self) { try cache.buffers(layer: 2, experts: [3]) }
        _ = try cache.buffers(layer: 2, experts: [0, 1])          // intact blobs still load
    }

    /// Greedy decode on the tiny 4-bit container with a 16-slot cache (so
    /// experts are evicted and prefetched): logits are bitwise identical with
    /// the lookahead off, measuring, or prefetching.
    @Test func lookaheadNeverChangesTheOutput() throws {
        let out = try ExpertCacheTests.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let gb = Double(try ExpertCacheTests.budget(slots: 16, container: out)) / 1_073_741_824
        let off = try QwenMetalModel(modelDir: out, cacheBudgetGB: gb)
        let measure = try QwenMetalModel(modelDir: out, cacheBudgetGB: gb)
        let prefetch = try QwenMetalModel(modelDir: out, cacheBudgetGB: gb)
        measure.lookahead = .measure
        prefetch.lookahead = .prefetch
        prefetch.lookaheadWidth = 4

        let tokens = [1, 5, 9, 42, 7, 99, 3, 17, 64, 2, 11, 30]
        let states = (0..<3).map { _ in QwenCPUModel.DecodeState() }
        for t in tokens {
            let a = try off.step([t], state: states[0])
            let b = try measure.step([t], state: states[1])
            let c = try prefetch.step([t], state: states[2])
            #expect(a == b, "measure changed the logits at token \(t)")
            #expect(a == c, "prefetch changed the logits at token \(t)")
        }
        let layers = off.config.numHiddenLayers
        #expect(measure.lookaheadStats.layers == tokens.count * (layers - 1))
        #expect(measure.lookaheadStats.guessed <= measure.lookaheadStats.picked)
        #expect(measure.lookaheadStats.coldGuessed <= measure.lookaheadStats.cold)
        #expect(off.lookaheadStats.layers == 0)
        let cache = try #require(prefetch.expertCache)
        #expect(cache.prefetchIssued > 0, "a 16-slot cache should have needed prefetches")
        #expect(cache.prefetchUsed <= cache.prefetchIssued)
    }
}
