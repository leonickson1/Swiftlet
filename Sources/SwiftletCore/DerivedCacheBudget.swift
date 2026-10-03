import Foundation
import Metal

/// The expert-cache budget a host can hold beside a model, derived from what
/// the machine offers and what the model keeps resident:
///
///     headroom(tokens) = min(working set, available) − resident dense
///                        − fixed state − KV(tokens) − margin
///
/// The default budget is `headroom(contextCapacity)` clamped to
/// `[floor, pool]`, so a host that can hold the whole expert pool beside a
/// full-length context gets the whole pool, and a host that cannot hold the
/// KV cache of a full-length context at all still gets a working cache. An
/// explicit `--cache-gb` is a ceiling on that pool: it is honoured up to
/// `headroom(minimumContextTokens)` — the host can hold it beside at least a
/// short context — and refused above it, naming the request, the ceiling and
/// the shortfall. One formula produces both numbers; only the KV horizon
/// differs.
///
/// Everything here is arithmetic on the two value types, so the policy is
/// testable without a GPU; `Host.sample(device:)` is the one platform call.
public struct ExpertCacheBudgetDerivation: Equatable, Sendable {
    /// What the host offers, sampled before the model's dense weights are
    /// copied resident so that the resident term is not paid twice.
    public struct Host: Equatable, Sendable {
        /// Bytes the GPU can keep resident at once: `recommendedMaxWorkingSetSize`
        /// on macOS, the process's jetsam allowance (`os_proc_available_memory`)
        /// on iOS.
        public var workingSetBytes: Int
        /// Physical memory of the machine; the plausibility floor for
        /// `availableBytes`.
        public var physicalMemoryBytes: Int
        /// Reclaimable memory right now (free, inactive and purgeable pages).
        /// File-backed pages — the page cache holding the container — count
        /// as reclaimable, so a model that was just read does not shrink its
        /// own budget. Nil when the kernel statistics could not be read.
        public var availableBytes: Int?

        public init(workingSetBytes: Int, physicalMemoryBytes: Int, availableBytes: Int?) {
            self.workingSetBytes = workingSetBytes
            self.physicalMemoryBytes = physicalMemoryBytes
            self.availableBytes = availableBytes
        }

        /// Reads the live host. `workingSetBytes` comes from the Metal device
        /// on macOS; the other two from `ProcessInfo` and `host_statistics64`.
        public static func sample(device: MTLDevice) -> Host {
            Host(
                workingSetBytes: workingSetBytes(device: device),
                physicalMemoryBytes: Int(clamping: ProcessInfo.processInfo.physicalMemory),
                availableBytes: sampleAvailableBytes()
            )
        }

        private static func workingSetBytes(device: MTLDevice) -> Int {
            #if os(macOS) || targetEnvironment(macCatalyst)
            return Int(clamping: device.recommendedMaxWorkingSetSize)
            #else
            // iOS exposes no recommended working set; the jetsam allowance
            // left to this process, read before the model loads, is the
            // number that decides whether an allocation survives. A zero
            // reading (simulator, entitlement missing) falls back to half of
            // physical memory, the historical rule of thumb.
            let allowance = Int(clamping: os_proc_available_memory())
            if allowance > 0 { return allowance }
            return Int(clamping: ProcessInfo.processInfo.physicalMemory / 2)
            #endif
        }

        /// Free + inactive + purgeable pages, in bytes. Inactive pages include
        /// the page cache, which is what makes a freshly read container not
        /// count against its own budget.
        static func sampleAvailableBytes() -> Int? {
            var stats = vm_statistics64()
            var count = mach_msg_type_number_t(
                MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
            let result = withUnsafeMutablePointer(to: &stats) { pointer in
                pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
                }
            }
            guard result == KERN_SUCCESS else { return nil }
            // vm_statistics64 counts kernel pages (16 KiB on Apple silicon),
            // so the page size is read from the host, not assumed.
            var pageSize: vm_size_t = 0
            guard host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS, pageSize > 0 else { return nil }
            let pages = Int(stats.free_count) + Int(stats.inactive_count) + Int(stats.purgeable_count)
            return pages * Int(pageSize)
        }
    }

    /// What the model keeps beside the expert cache, priced from the loaded
    /// checkpoint rather than from a per-model constant.
    public struct Model: Equatable, Sendable {
        /// Bytes every resident dense linear occupies (`MetalShardStore.residentBytes`).
        public var residentDenseBytes: Int
        /// Bytes one context token costs in KV, as the runtime allocates it
        /// (f32 K and V rows on every full-attention layer).
        public var kvBytesPerToken: Int
        /// Tokens the KV cache may grow to: `InferenceModel.contextCapacity`.
        public var contextCapacity: Int
        /// State that does not grow with the context: DeltaNet recurrent state
        /// and convolution history.
        public var fixedStateBytes: Int
        /// Bytes per expert slot in the container.
        public var expertStride: Int
        /// Routed experts in the container (layers × experts per layer); the
        /// cache can never usefully hold more.
        public var expertSlots: Int
        /// Experts one decode token touches (layers × top-k); the smallest
        /// cache that holds one token's working set.
        public var expertFetchesPerToken: Int

        public init(
            residentDenseBytes: Int, kvBytesPerToken: Int, contextCapacity: Int,
            fixedStateBytes: Int, expertStride: Int, expertSlots: Int, expertFetchesPerToken: Int
        ) {
            self.residentDenseBytes = residentDenseBytes
            self.kvBytesPerToken = kvBytesPerToken
            self.contextCapacity = contextCapacity
            self.fixedStateBytes = fixedStateBytes
            self.expertStride = expertStride
            self.expertSlots = expertSlots
            self.expertFetchesPerToken = expertFetchesPerToken
        }
    }

    public enum Error: Swift.Error, Equatable, CustomStringConvertible {
        /// The requested budget exceeds what the host can hold beside the
        /// model and a `minimumContextTokens` context.
        case budgetExceedsHost(requestedBytes: Int, ceilingBytes: Int, shortBytes: Int)
        /// Even the floor budget does not fit beside the model.
        case hostCannotHoldModel(neededBytes: Int, workingSetBytes: Int, shortBytes: Int)
        /// A container shape the formula cannot price.
        case invalidModelShape(String)
        /// A requested budget that is not a finite, non-negative number.
        case invalidBudget(String)

        public var description: String {
            switch self {
            case let .budgetExceedsHost(requested, ceiling, short):
                return "expert cache budget \(gb(requested)) exceeds what this host can hold "
                    + "beside the model, \(gb(ceiling)); short by \(gb(short)) "
                    + "(pass --cache-gb \(gbNumber(ceiling)) or less, or omit it for the derived default)"
            case let .hostCannotHoldModel(needed, workingSet, short):
                return "this host cannot hold the model: \(gb(needed)) needed for the resident "
                    + "dense weights, the smallest expert cache, a minimum context and the margin, "
                    + "but the working set is \(gb(workingSet)); short by \(gb(short))"
            case .invalidModelShape(let what):
                return "expert cache budget cannot be derived: \(what)"
            case .invalidBudget(let what):
                return "expert cache budget \(what) is not a finite, non-negative number of GB"
            }
        }
    }

    /// The smallest context an explicit budget must leave room for. A host
    /// is refused an explicit `--cache-gb` only when the cache would not fit
    /// beside the model and this many tokens of KV; the derived default
    /// reserves KV for `contextCapacity` instead.
    public static let minimumContextTokens = 4_096
    /// The cache's own minimum (`ExpertCacheBudget.slotCapacity`).
    public static let minimumSlots = 16
    public static let defaultMarginFraction = 0.10
    public static let defaultMinimumMarginBytes = 1 << 30

    public let host: Host
    public let model: Model
    /// Headroom kept for scratch buffers, the tokenizer, and the OS:
    /// `marginFraction` of the working set, at least `minimumMarginBytes`.
    public let marginBytes: Int
    /// `availableBytes` when it was measured and plausible (at least 2 % of
    /// physical memory); nil when unmeasured or implausible, in which case
    /// the working set alone bounds the default.
    public let plausibleAvailableBytes: Int?
    /// Bytes the whole routed-expert pool occupies.
    public let poolBytes: Int
    /// The smallest useful cache: one decode token's working set, never
    /// fewer than `minimumSlots` slots, never more than the pool.
    public let floorBytes: Int
    /// KV bytes reserved for a context of `contextCapacity` tokens.
    public let kvReserveBytes: Int
    /// `headroom(contextCapacity)` before clamping; may be negative.
    public let headroomBytes: Int
    /// The default budget: `headroomBytes` clamped to `[floorBytes, poolBytes]`.
    public let derivedBytes: Int
    /// The most an explicit budget may ask for: `headroom(minimumContextTokens)`
    /// against the working set alone, capped at the pool.
    public let ceilingBytes: Int

    public init(
        host: Host, model: Model,
        marginFraction: Double = ExpertCacheBudgetDerivation.defaultMarginFraction,
        minimumMarginBytes: Int = ExpertCacheBudgetDerivation.defaultMinimumMarginBytes
    ) throws {
        guard model.expertStride > 0 else { throw Error.invalidModelShape("expert stride is 0") }
        guard model.expertSlots > 0 else { throw Error.invalidModelShape("container holds no experts") }
        guard model.kvBytesPerToken >= 0, model.contextCapacity >= 0 else {
            throw Error.invalidModelShape("negative KV size")
        }
        guard host.workingSetBytes > 0 else { throw Error.invalidModelShape("host working set is 0") }
        let (pool, poolOverflow) = model.expertStride.multipliedReportingOverflow(by: model.expertSlots)
        guard !poolOverflow else { throw Error.invalidModelShape("expert pool size overflows") }
        let (kvReserve, kvOverflow) = model.kvBytesPerToken.multipliedReportingOverflow(by: model.contextCapacity)
        guard !kvOverflow else { throw Error.invalidModelShape("KV reserve overflows") }

        self.host = host
        self.model = model
        poolBytes = pool
        kvReserveBytes = kvReserve
        let marginFromFraction = Int((Double(host.workingSetBytes) * max(0, marginFraction)).rounded(.up))
        marginBytes = max(max(0, minimumMarginBytes), marginFromFraction)
        floorBytes = min(pool, max(Self.minimumSlots, model.expertFetchesPerToken) * model.expertStride)

        if let available = host.availableBytes, host.physicalMemoryBytes > 0,
           available >= host.physicalMemoryBytes / 50 {
            plausibleAvailableBytes = available
        } else {
            plausibleAvailableBytes = nil
        }

        // The default prices KV at the full context and bows to what is
        // reclaimable now; the ceiling prices KV at the minimum context
        // against the GPU's hard limit only, because an explicit request is
        // the user's decision about the rest of the machine.
        let offered = min(host.workingSetBytes, plausibleAvailableBytes ?? host.workingSetBytes)
        let fixed = model.residentDenseBytes + model.fixedStateBytes + marginBytes
        headroomBytes = offered - fixed - kvReserve
        derivedBytes = min(pool, max(floorBytes, headroomBytes))
        let minimumKV = model.kvBytesPerToken * min(model.contextCapacity, Self.minimumContextTokens)
        ceilingBytes = min(pool, host.workingSetBytes - fixed - minimumKV)

        if ceilingBytes < floorBytes {
            let needed = fixed + minimumKV + floorBytes
            throw Error.hostCannotHoldModel(
                neededBytes: needed, workingSetBytes: host.workingSetBytes,
                shortBytes: needed - host.workingSetBytes)
        }
    }

    /// The budget to run at. Nil asks for the derived default. An explicit
    /// request above the pool is cut to the pool (more can never be used);
    /// what would actually be allocated must then fit under `ceilingBytes`.
    public func admit(requestedBytes: Int?) throws -> Int {
        guard let requested = requestedBytes else { return derivedBytes }
        guard requested >= 0 else { throw Error.invalidBudget("\(requested) B") }
        let allocated = min(requested, poolBytes)
        guard allocated <= ceilingBytes else {
            throw Error.budgetExceedsHost(
                requestedBytes: requested, ceilingBytes: ceilingBytes,
                shortBytes: allocated - ceilingBytes)
        }
        return allocated
    }

    /// `admit` for a budget in GB, the `--cache-gb` unit.
    public func admit(requestedGB: Double?) throws -> Int {
        guard let gb = requestedGB else { return derivedBytes }
        guard gb.isFinite, gb >= 0, let bytes = Int(exactly: (gb * 1_073_741_824).rounded(.down)) else {
            throw Error.invalidBudget("\(gb)")
        }
        return try admit(requestedBytes: bytes)
    }

    /// How many context tokens the host can hold beside a cache of `budgetBytes`
    /// under the same accounting (offered − resident − fixed − margin − cache),
    /// capped at `contextCapacity`. What the prompt path can borrow.
    public func contextTokensHoldable(besideCacheBytes budgetBytes: Int) -> Int {
        guard model.kvBytesPerToken > 0 else { return model.contextCapacity }
        let offered = min(host.workingSetBytes, plausibleAvailableBytes ?? host.workingSetBytes)
        let left = offered - model.residentDenseBytes - model.fixedStateBytes - marginBytes - budgetBytes
        guard left > 0 else { return 0 }
        return min(model.contextCapacity, left / model.kvBytesPerToken)
    }

    /// One line for the startup log: the derivation with every term named,
    /// and what `budgetBytes` (derived or explicit) leaves for the context.
    public func summary(budgetBytes: Int, requestedGB: Double?) -> String {
        var terms = "working set \(gb(host.workingSetBytes))"
        if let available = plausibleAvailableBytes {
            terms += " (available \(gb(available)))"
        } else if host.availableBytes != nil {
            terms += " (available reading implausible, ignored)"
        } else {
            terms += " (available unmeasured)"
        }
        terms += " - resident dense \(gb(model.residentDenseBytes))"
        terms += " - KV \(gb(kvReserveBytes)) at \(model.contextCapacity) tokens"
        terms += " - fixed state \(gb(model.fixedStateBytes))"
        terms += " - margin \(gb(marginBytes))"
        terms += " = \(gb(headroomBytes))"
        let clamp: String
        if headroomBytes > poolBytes {
            clamp = ", clamped to the \(gb(poolBytes)) pool"
        } else if headroomBytes < floorBytes {
            clamp = ", raised to the \(gb(floorBytes)) floor"
        } else {
            clamp = " (pool \(gb(poolBytes)), floor \(gb(floorBytes)))"
        }
        let chosen: String
        if let requestedGB {
            let note = budgetBytes < Int((requestedGB * 1_073_741_824).rounded(.down))
                ? " cut to the pool" : ""
            chosen = "expert cache budget \(gb(budgetBytes)) (--cache-gb \(gbNumber(requestedGB))\(note); "
                + "ceiling \(gb(ceilingBytes)); derived default \(gb(derivedBytes)))"
        } else {
            chosen = "expert cache budget \(gb(budgetBytes)) (derived)"
        }
        return chosen + ": " + terms + clamp
            + "; KV headroom beside it \(contextTokensHoldable(besideCacheBytes: budgetBytes)) tokens"
    }
}

private func gb(_ bytes: Int) -> String {
    String(format: "%.2f GB", Double(bytes) / 1_073_741_824)
}

private func gbNumber(_ bytes: Int) -> String {
    gbNumber(Double(bytes) / 1_073_741_824)
}

private func gbNumber(_ value: Double) -> String {
    let text = String(format: "%.2f", value)
    var trimmed = text
    while trimmed.contains("."), trimmed.hasSuffix("0") { trimmed.removeLast() }
    if trimmed.hasSuffix(".") { trimmed.removeLast() }
    return trimmed
}
