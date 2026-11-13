import Foundation

/// Per-fetch observation sink for ``ExpertCache``, off unless attached.
///
/// The cache's hit and miss counters say how often the pool answered, not
/// what a miss cost. A miss is one of two very different things: a pread the
/// macOS page cache answers from RAM (a syscall and a 1.7 MiB copy) or a
/// pread that waits on the device. Telling them apart is what makes a cache
/// experiment honest, so with a sink attached every `buffers` call records
/// the layer, the routed experts in request order, and per expert whether it
/// hit, how long its pread took, and how much of the blob the page cache
/// already held the instant before the read was issued (a read-only mapping
/// plus mincore(2), neither of which faults a page in).
///
/// Cost when attached: one clock read per call, two clock reads and one
/// mincore per miss. Cost when nil: nothing on the hot path. The CLI attaches
/// a sink for `--trace-steps`; see docs/CACHE_AB.md.
public final class ExpertFetchTrace {
    public enum Outcome: Equatable, Sendable {
        /// Resident in the pool: no syscall, no copy.
        case hit
        /// Filled from the container. `preadSeconds` is this blob's own
        /// pread wall time (a batch reads concurrently, so a batch's sum
        /// exceeds its wall time). `residentFraction` is the share of the
        /// blob's pages the page cache held just before the read: 1 means
        /// every byte came from RAM, 0 means every byte came from the
        /// device, nil means the probe was unavailable.
        case miss(preadSeconds: Double, residentFraction: Double?)
    }

    public struct FetchRecord: Equatable, Sendable {
        public let layer: Int
        public let experts: [Int]
        /// Parallel to `experts`.
        public let outcomes: [Outcome]
        /// Wall time of the whole `buffers` call: policy, probes and reads.
        public let seconds: Double

        public init(layer: Int, experts: [Int], outcomes: [Outcome], seconds: Double) {
            self.layer = layer
            self.experts = experts
            self.outcomes = outcomes
            self.seconds = seconds
        }
    }

    public private(set) var records: [FetchRecord] = []

    public init() {}

    func append(_ record: FetchRecord) { records.append(record) }

    /// Everything recorded since the last drain, in call order. The caller
    /// (one decode step at a time) owns the boundary between steps.
    public func drain() -> [FetchRecord] {
        let out = records
        records.removeAll(keepingCapacity: true)
        return out
    }
}

extension ExpertFetchTrace {
    /// One step's fetches folded by where they were served from.
    public struct StepSummary: Equatable, Sendable {
        /// Upper edges in milliseconds of the pread histogram buckets; the
        /// last bucket is open. A page cache pread of a 1.7 MiB blob lands
        /// in the first two or three buckets, a device read further right.
        public static let histogramEdgesMs: [Double] = [0.25, 0.5, 1, 2, 4, 8, 16]

        public var calls = 0
        public var fetches = 0
        public var hits = 0
        public var misses = 0
        /// Misses whose every page was already resident: a copy, no I/O.
        public var missesFromPageCache = 0
        /// Misses with at least one page not resident: real device I/O.
        public var missesFromDevice = 0
        /// Misses the probe could not classify.
        public var missesUnprobed = 0
        /// Sum of per-blob pread wall time over the step's misses.
        public var preadSeconds = 0.0
        public var preadMaxSeconds = 0.0
        /// Sum of residentFraction over probed misses.
        public var residentFractionSum = 0.0
        /// Pread wall time histogram, buckets bounded by `histogramEdgesMs`.
        public var preadHistogram = [Int](repeating: 0, count: histogramEdgesMs.count + 1)
        /// Sum of `FetchRecord.seconds`.
        public var fetchSeconds = 0.0

        public init() {}

        public init(records: [FetchRecord]) {
            self.init()
            for record in records { add(record) }
        }

        public mutating func add(_ record: FetchRecord) {
            calls += 1
            fetchSeconds += record.seconds
            for outcome in record.outcomes {
                fetches += 1
                switch outcome {
                case .hit:
                    hits += 1
                case .miss(let seconds, let resident):
                    misses += 1
                    preadSeconds += seconds
                    preadMaxSeconds = max(preadMaxSeconds, seconds)
                    preadHistogram[Self.bucket(seconds: seconds)] += 1
                    if let resident {
                        residentFractionSum += resident
                        if resident >= 1 { missesFromPageCache += 1 } else { missesFromDevice += 1 }
                    } else {
                        missesUnprobed += 1
                    }
                }
            }
        }

        /// Mean page cache residency over probed misses, nil when none.
        public var residentFractionMean: Double? {
            let probed = misses - missesUnprobed
            return probed > 0 ? residentFractionSum / Double(probed) : nil
        }

        public static func bucket(seconds: Double) -> Int {
            let ms = seconds * 1000
            return histogramEdgesMs.firstIndex { ms < $0 } ?? histogramEdgesMs.count
        }
    }
}
