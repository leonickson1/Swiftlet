import Foundation
import Metal
import Testing
@testable import SwiftletCore

/// The fetch trace is the instrument behind docs/CACHE_AB.md. It must agree
/// with the cache's own counters call for call, classify every miss by
/// where its bytes came from, and leave the untraced path untouched.
@Suite struct ExpertFetchTraceTests {
    @Test func recordsAgreeWithTheCountersCallForCall() throws {
        let out = try ExpertCacheTests.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = try ExpertCache(
            containerDir: out, device: device,
            budgetBytes: ExpertCacheTests.budget(slots: 16, container: out))
        let sink = ExpertFetchTrace()
        cache.trace = sink
        let layers = cache.layout.layerCount
        let expertsPerLayer = cache.layout.expertCount

        var rng = UInt64(0x2545F4914F6CDD1D)
        func next(_ n: Int) -> Int {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return Int((rng >> 33) % UInt64(n))
        }
        for call in 0..<300 {
            let layer = next(layers)
            var batch: [Int] = []
            let want = 1 + next(4)
            while batch.count < want {
                let e = next(expertsPerLayer)
                if !batch.contains(e) { batch.append(e) }
            }
            // A hit needs the expert resident before the call. The converse
            // does not hold: an earlier miss in the same batch may evict a
            // later member that was resident (only processed members are
            // protected), so a miss on a resident expert is legal exactly
            // when a miss preceded it in the batch.
            let residentBefore = batch.map { cache.residentSlot(layer: layer, expert: $0) != nil }
            let hits0 = cache.hits, misses0 = cache.misses
            _ = try cache.buffers(layer: layer, experts: batch)
            let records = sink.drain()
            try #require(records.count == 1, "call \(call): one record per buffers call")
            let record = records[0]
            #expect(record.layer == layer && record.experts == batch)
            #expect(record.outcomes.count == batch.count)
            #expect(record.seconds >= 0)
            var hits = 0, misses = 0
            for (i, outcome) in record.outcomes.enumerated() {
                switch outcome {
                case .hit:
                    hits += 1
                    #expect(residentBefore[i], "call \(call): hit recorded for a non-resident expert")
                case .miss(let seconds, let resident):
                    if residentBefore[i] {
                        #expect(misses > 0, "call \(call): miss on a resident expert with no earlier miss to evict it")
                    }
                    misses += 1
                    #expect(seconds >= 0)
                    if let resident { #expect(resident >= 0 && resident <= 1) }
                }
            }
            #expect(hits == cache.hits - hits0 && misses == cache.misses - misses0,
                    "call \(call): record disagrees with the counters")
        }
        #expect(cache.misses > cache.slotCount, "stream never evicted; the policy was not exercised")
        #expect(cache.hits > 0)
        #expect(sink.drain().isEmpty)
    }

    @Test func detachedCacheRecordsNothing() throws {
        let out = try ExpertCacheTests.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = try ExpertCache(
            containerDir: out, device: device,
            budgetBytes: ExpertCacheTests.budget(slots: 16, container: out))
        let sink = ExpertFetchTrace()
        cache.trace = sink
        _ = try cache.buffers(layer: 0, experts: [0, 1])
        cache.trace = nil
        _ = try cache.buffers(layer: 0, experts: [2, 3])
        #expect(sink.records.count == 1, "only the attached call is recorded")
        #expect(cache.hits + cache.misses == 4, "the untraced call still counts")
    }

    @Test func drainHandsBackEverythingOnce() {
        let sink = ExpertFetchTrace()
        sink.append(.init(layer: 0, experts: [1], outcomes: [.hit], seconds: 0))
        sink.append(.init(layer: 1, experts: [2],
                          outcomes: [.miss(preadSeconds: 0.001, residentFraction: 1)], seconds: 0.001))
        #expect(sink.drain().count == 2)
        #expect(sink.drain().isEmpty)
        #expect(sink.records.isEmpty)
    }

    @Test func stepSummaryFoldsMissesByTier() {
        let records = [
            ExpertFetchTrace.FetchRecord(layer: 0, experts: [1, 2, 3, 4], outcomes: [
                .hit,
                .miss(preadSeconds: 0.0001, residentFraction: 1),    // page cache, 0.1 ms
                .miss(preadSeconds: 0.003, residentFraction: 0.5),   // device, 3 ms
                .miss(preadSeconds: 0.020, residentFraction: nil),   // unprobed, 20 ms
            ], seconds: 0.025),
            ExpertFetchTrace.FetchRecord(layer: 1, experts: [7], outcomes: [.hit], seconds: 0.00001),
        ]
        let s = ExpertFetchTrace.StepSummary(records: records)
        #expect(s.calls == 2 && s.fetches == 5 && s.hits == 2 && s.misses == 3)
        #expect(s.missesFromPageCache == 1 && s.missesFromDevice == 1 && s.missesUnprobed == 1)
        #expect(abs(s.preadSeconds - 0.0231) < 1e-9)
        #expect(s.preadMaxSeconds == 0.020)
        #expect(s.residentFractionMean == 0.75)
        #expect(s.preadHistogram == [1, 0, 0, 0, 1, 0, 0, 1])
        #expect(abs(s.fetchSeconds - 0.02501) < 1e-9)
        #expect(ExpertFetchTrace.StepSummary().residentFractionMean == nil)
        #expect(ExpertFetchTrace.StepSummary.bucket(seconds: 0) == 0)
        #expect(ExpertFetchTrace.StepSummary.bucket(seconds: 0.016) == 7)
        #expect(ExpertFetchTrace.StepSummary.bucket(seconds: 0.00025) == 1)
    }

    @Test func pageCacheProbeReportsWhatAReadLeavesResident() throws {
        let out = try ExpertCacheTests.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let reader = try QpackExpertReader(containerDir: out)
        #expect(reader.pageCacheProbeAvailable)
        let before = try #require(reader.pageCacheResidentFraction(layer: 0, expert: 1))
        #expect(before >= 0 && before <= 1)
        let buf = UnsafeMutableRawPointer.allocate(byteCount: reader.layout.expertStride, alignment: 16)
        defer { buf.deallocate() }
        try reader.readExpert(layer: 0, expert: 1, into: buf)
        #expect(reader.pageCacheResidentFraction(layer: 0, expert: 1) == 1,
                "every page of a blob just read through pread must be resident")
        #expect(reader.pageCacheResidentFraction(layer: 0, expert: reader.layout.expertCount) == nil,
                "an out of range expert is refused, not probed past the file")
        #expect(reader.pageCacheResidentFraction(layer: reader.layout.layerCount, expert: 0) == nil)
        #expect(reader.pageCacheResidentFraction(layer: 0, expert: -1) == nil)
    }

    @Test func timedReadsReturnTheSameBytesPlusOneTimingPerRequest() throws {
        let out = try ExpertCacheTests.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let reader = try QpackExpertReader(containerDir: out)
        let stride = reader.layout.expertStride
        let n = min(4, reader.layout.expertCount)
        let plain = (0..<n).map { _ in UnsafeMutableRawPointer.allocate(byteCount: stride, alignment: 16) }
        let timed = (0..<n).map { _ in UnsafeMutableRawPointer.allocate(byteCount: stride, alignment: 16) }
        defer { (plain + timed).forEach { $0.deallocate() } }
        try reader.readExperts(layer: 0, (0..<n).map { (expert: $0, into: plain[$0]) })
        let timings = try reader.readExpertsTimed(layer: 0, (0..<n).map { (expert: $0, into: timed[$0]) })
        #expect(timings.count == n)
        #expect(timings.allSatisfy { $0 >= 0 })
        for i in 0..<n {
            #expect(memcmp(plain[i], timed[i], stride) == 0, "expert \(i): timed read returned different bytes")
        }
        #expect(try reader.readExpertsTimed(layer: 0, []).isEmpty)
        let single = try reader.readExpertsTimed(layer: 0, [(expert: 0, into: timed[0])])
        #expect(single.count == 1 && single[0] >= 0)
    }

    /// --no-page-cache changes where a miss's bytes come from, never what
    /// they are, whether set before the layer files open or after.
    @Test func bypassingThePageCacheReadsTheSameBytes() throws {
        let out = try ExpertCacheTests.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let plainReader = try QpackExpertReader(containerDir: out)
        let earlyBypass = try QpackExpertReader(containerDir: out)
        earlyBypass.bypassPageCache = true
        let lateBypass = try QpackExpertReader(containerDir: out)
        let stride = plainReader.layout.expertStride
        let bufs = (0..<4).map { _ in UnsafeMutableRawPointer.allocate(byteCount: stride, alignment: 16) }
        defer { bufs.forEach { $0.deallocate() } }
        for layer in 0..<plainReader.layout.layerCount {
            for expert in 0..<plainReader.layout.expertCount {
                try plainReader.readExpert(layer: layer, expert: expert, into: bufs[0])
                try earlyBypass.readExpert(layer: layer, expert: expert, into: bufs[1])
                try lateBypass.readExpert(layer: layer, expert: expert, into: bufs[2])
                lateBypass.bypassPageCache = true  // now applies to an open file
                try lateBypass.readExpert(layer: layer, expert: expert, into: bufs[3])
                lateBypass.bypassPageCache = false
                for i in 1..<4 {
                    #expect(memcmp(bufs[0], bufs[i], stride) == 0, "layer \(layer) expert \(expert) reader \(i)")
                }
            }
        }
        let device = try #require(MTLCreateSystemDefaultDevice())
        let cache = try ExpertCache(
            containerDir: out, device: device,
            budgetBytes: ExpertCacheTests.budget(slots: 16, container: out))
        #expect(!cache.bypassPageCache)
        cache.bypassPageCache = true
        #expect(cache.bypassPageCache)
    }
}
