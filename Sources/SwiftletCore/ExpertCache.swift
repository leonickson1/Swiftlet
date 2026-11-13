import Foundation
import Metal

/// Physical-allocation budget used by ``ExpertCache``. Metal may round a
/// buffer's logical length up to a larger allocation, so the cache charges the
/// resource's actual `allocatedSize` before retaining it.
struct ExpertCacheBudget {
    let limitBytes: Int
    private(set) var allocatedBytes = 0

    static func slotCapacity(
        limitBytes: Int, stride: Int, totalSlots: Int, minimumSlots: Int = 16
    ) -> Int? {
        guard limitBytes >= 0, stride > 0, totalSlots > 0, minimumSlots >= 0 else {
            return nil
        }
        let capacity = min(limitBytes / stride, totalSlots)
        return capacity >= min(minimumSlots, totalSlots) ? capacity : nil
    }

    mutating func reserve(_ bytes: Int) -> Bool {
        guard bytes > 0, allocatedBytes <= limitBytes,
              bytes <= limitBytes - allocatedBytes else { return false }
        allocatedBytes += bytes
        return true
    }
}

/// Bounded expert-blob cache over a .qpack container: a lazily allocated pool
/// of GPU-shared buffers, filled by single preads, with LRU eviction by
/// default (`policy`; LFU with a recency tie-break, TurboFieldfare's policy,
/// remains selectable). Both logical slot
/// capacity and Metal's physical allocation size are bounded by `budgetBytes`,
/// so the working set can never thrash the machine.
///
/// S3c: a batch resolves every hit and every victim first (the policy is
/// sequential by definition -- each victim choice excludes the batch's
/// earlier members), then issues the batch's preads together. Victim
/// selection scans two flat per-slot arrays instead of two dictionaries;
/// the choice is identical (minimum (frequency, last use), lowest slot on
/// ties), only cheaper.
public final class ExpertCache {
    /// Which resident expert a miss evicts.
    public enum EvictionPolicy: String, Sendable {
        /// Fewest accesses ever (frequency history survives eviction), most
        /// recent use breaking ties. The default until LRU measured better.
        case lfu
        /// Least recently used. The default: in a session of unrelated
        /// prompts the LFU history keeps earlier prompts' favourites
        /// resident. On the docs/CACHE_SUITE.md suite at 8 GiB, LRU hit
        /// 93.2% against LFU's 77.8% on Qwen3.6-35B (Belady 97.1%) and 84.9%
        /// against 55.5% on Qwen3-Next-80B, and decoded 16-35% faster on the
        /// 35B.
        case lru
    }

    /// Takes effect from the next miss; resident keys and history are kept.
    public var policy: EvictionPolicy = .lru

    let reader: QpackExpertReader
    public let stride: Int
    private let device: MTLDevice

    private var slots: [MTLBuffer] = []
    private var slotKey: [Int64] = []          // key occupying each slot, -1 free
    private var keyToSlot: [Int64: Int] = [:]
    /// Access frequency per key, resident or not (LFU with history).
    private var freq: [Int64: Int] = [:]
    /// Per-slot mirror of `freq[slotKey[s]]` for resident keys, and the
    /// tick of each slot's last use -- the two numbers the victim scan reads.
    private var slotFreq: [Int] = []
    private var slotLastUse: [Int] = []
    private var tick = 0

    public private(set) var hits = 0
    public private(set) var misses = 0
    /// Optional per-fetch observer (docs/CACHE_AB.md). Nil, the default,
    /// keeps `buffers` exactly as it was; attached, each call appends one
    /// record with per-expert hit / miss outcomes, pread timing and the
    /// page cache's residency of each missed blob just before its read.
    public var trace: ExpertFetchTrace?

    private var maxSlots: Int
    private var physicalBudget: ExpertCacheBudget

    init(containerDir: URL, device: MTLDevice, budgetBytes: Int) throws {
        self.device = device
        reader = try QpackExpertReader(containerDir: containerDir)
        stride = reader.layout.expertStride
        guard stride > 0, !reader.layout.sections.isEmpty else {
            throw Checkpoint.Error.badShape("corrupt container: empty expert layout (re-download the model)")
        }
        let (total, overflow) = reader.layout.expertCount.multipliedReportingOverflow(
            by: reader.layout.layerCount
        )
        guard !overflow, total > 0 else {
            throw Checkpoint.Error.badShape("corrupt container: invalid expert count")
        }
        guard let capacity = ExpertCacheBudget.slotCapacity(
            limitBytes: budgetBytes, stride: stride, totalSlots: total
        ) else {
            let availableSlots = budgetBytes >= 0 ? budgetBytes / stride : 0
            throw Checkpoint.Error.badShape(
                "expert cache budget fits \(availableSlots) logical slots; "
                    + "at least \(min(16, total)) required"
            )
        }
        maxSlots = capacity
        physicalBudget = ExpertCacheBudget(limitBytes: budgetBytes)
        // Slots allocate lazily on demand (memory grows with use, never past
        // the budget) — important on iOS where an up-front multi-GB allocation
        // invites jetsam before the model even runs.
    }

    public var slotCount: Int { maxSlots }
    public var allocatedSlots: Int { slots.count }
    public var logicalBytes: Int { slots.count * stride }
    public var allocatedBytes: Int { physicalBudget.allocatedBytes }
    public var budgetBytes: Int { physicalBudget.limitBytes }
    public var layout: Qpack.Layout { reader.layout }
    /// Misses read past the page cache (QpackExpertReader.bypassPageCache).
    public var bypassPageCache: Bool {
        get { reader.bypassPageCache }
        set { reader.bypassPageCache = newValue }
    }
    /// Whether the fetch trace can classify misses by page cache residency.
    public var pageCacheProbeAvailable: Bool { reader.pageCacheProbeAvailable }

    private func key(_ layer: Int, _ expert: Int) -> Int64 {
        Int64(layer) << 32 | Int64(expert)
    }

    /// Test hook: the slot holding (layer, expert), or nil when not resident.
    func residentSlot(layer: Int, expert: Int) -> Int? {
        keyToSlot[key(layer, expert)]
    }

    /// Buffers for a batch of experts in one layer. The whole batch is
    /// resident simultaneously (a member of the batch is never evicted to make
    /// room for another member). Misses are read into their slots together
    /// (concurrent preads when there is more than one); a failed read leaves
    /// its slot free rather than claiming a key whose bytes never landed.
    func buffers(layer: Int, experts: [Int]) throws -> [MTLBuffer] {
        let trace = self.trace
        let start = trace == nil ? 0 : ProcessInfo.processInfo.systemUptime
        tick += 1
        var protectedSlots = Set<Int>()
        var result: [MTLBuffer] = []
        result.reserveCapacity(experts.count)
        var fills: [(slot: Int, expert: Int)] = []
        /// Parallel to `experts`, only kept while tracing.
        var missed: [Bool] = []
        if trace != nil { missed.reserveCapacity(experts.count) }

        for e in experts {
            let k = key(layer, e)
            let f = freq[k, default: 0] + 1
            freq[k] = f
            if let s = keyToSlot[k] {
                hits += 1
                slotLastUse[s] = tick
                slotFreq[s] = f
                protectedSlots.insert(s)
                result.append(slots[s])
                if trace != nil { missed.append(false) }
                continue
            }
            misses += 1
            let s = try slotForFill(excluding: protectedSlots)
            if slotKey[s] >= 0 { keyToSlot.removeValue(forKey: slotKey[s]) }
            slotKey[s] = k
            keyToSlot[k] = s
            slotLastUse[s] = tick
            slotFreq[s] = f
            protectedSlots.insert(s)
            result.append(slots[s])
            if trace != nil { missed.append(true) }
            fills.append((slot: s, expert: e))
        }

        var preadSeconds: [Double] = []
        var resident: [Double?] = []
        if !fills.isEmpty {
            if trace != nil {
                // Probe before the read; afterwards every page is resident.
                resident = fills.map { reader.pageCacheResidentFraction(layer: layer, expert: $0.expert) }
            }
            do {
                let requests = fills.map { (expert: $0.expert, into: slots[$0.slot].contents()) }
                if trace != nil {
                    preadSeconds = try reader.readExpertsTimed(layer: layer, requests)
                } else {
                    try reader.readExperts(layer: layer, requests)
                }
            } catch {
                // Nothing in the batch is trustworthy after a short read:
                // release every slot this batch filled so no key points at
                // bytes that never arrived.
                for fill in fills {
                    keyToSlot.removeValue(forKey: slotKey[fill.slot])
                    slotKey[fill.slot] = -1
                    slotFreq[fill.slot] = 0
                }
                throw error
            }
        }

        if let trace {
            var outcomes: [ExpertFetchTrace.Outcome] = []
            outcomes.reserveCapacity(experts.count)
            var fill = 0
            for wasMiss in missed {
                if wasMiss {
                    outcomes.append(.miss(preadSeconds: preadSeconds[fill], residentFraction: resident[fill]))
                    fill += 1
                } else {
                    outcomes.append(.hit)
                }
            }
            trace.append(ExpertFetchTrace.FetchRecord(
                layer: layer, experts: experts, outcomes: outcomes,
                seconds: ProcessInfo.processInfo.systemUptime - start))
        }
        return result
    }

    /// Victim per `policy` (LFU with recency tie-break, or LRU);
    /// grow-on-demand, then free slots, then eviction.
    private func slotForFill(excluding: Set<Int>) throws -> Int {
        if slots.count < maxSlots {
            guard let b = device.makeBuffer(length: stride, options: .storageModeShared) else {
                // Treat an allocator refusal as the discovered physical cap;
                // repeated retries on every cache miss only make pressure worse.
                maxSlots = slots.count
                return try existingSlot(excluding: excluding)
            }
            // Never undercharge a resource even if a Metal implementation
            // reports an unexpected value smaller than the requested length.
            let charge = max(stride, b.allocatedSize)
            if physicalBudget.reserve(charge) {
                slots.append(b)
                slotKey.append(-1)
                slotFreq.append(0)
                slotLastUse.append(0)
                return slots.count - 1
            }
            // Resource rounding made the physical capacity smaller than the
            // logical estimate. Remember the discovered cap so every miss
            // does not allocate and immediately discard another buffer.
            maxSlots = slots.count
        }
        return try existingSlot(excluding: excluding)
    }

    private func existingSlot(excluding: Set<Int>) throws -> Int {
        if let free = slotKey.firstIndex(of: -1) { return free }
        var victim = -1
        var victimFreq = Int.max
        var victimUse = Int.max
        let lru = policy == .lru
        for s in 0..<slots.count where !excluding.contains(s) {
            // LRU ranks every slot at the same frequency, so the recency
            // tie-break alone decides (lowest slot on full ties, as LFU).
            let f = lru ? 0 : slotFreq[s]
            let u = slotLastUse[s]
            if f < victimFreq || (f == victimFreq && u < victimUse) {
                victimFreq = f
                victimUse = u
                victim = s
            }
        }
        guard victim >= 0 else {
            throw Checkpoint.Error.badShape(
                "expert cache budget fits \(slots.count) physical slots; "
                    + "batch requires at least \(excluding.count + 1)"
            )
        }
        return victim
    }
}
