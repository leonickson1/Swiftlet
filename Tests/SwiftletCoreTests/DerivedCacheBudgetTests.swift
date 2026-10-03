import Foundation
import Metal
import Testing
@testable import SwiftletCore

/// The host-derived expert-cache budget: the formula on several host shapes
/// (pure arithmetic, no GPU), and on a real Metal device the T13 rule that
/// the budget changes only speed, never output.
@Suite struct DerivedCacheBudgetTests {
    typealias Derivation = ExpertCacheBudgetDerivation

    static let GiB = 1 << 30
    static let MiB = 1 << 20

    /// Qwen3.6-35B-A3B 4-bit as measured on an M4 Max (VALIDATION_LOG
    /// 2026-09-28): 1,173 resident linears, 10 full-attention layers with
    /// 2 KV heads × 256 at f32, 262,144 trained positions, 40 × 256 experts
    /// of 1,769,472 B, 8 routed experts per token over 40 layers.
    static let qwen35B = Derivation.Model(
        residentDenseBytes: 1_101_009_920,
        kvBytesPerToken: 10 * 2 * 2 * 256 * 4,
        contextCapacity: 262_144,
        fixedStateBytes: 60 * Self.MiB,
        expertStride: 1_769_472,
        expertSlots: 40 * 256,
        expertFetchesPerToken: 40 * 8
    )
    static let pool35B = 1_769_472 * 40 * 256          // 16.875 GiB
    static let floor35B = 320 * 1_769_472               // 540 MiB, one token's working set

    static func host(workingSetGiB: Double, physicalGiB: Double, availableGiB: Double?) -> Derivation.Host {
        Derivation.Host(
            workingSetBytes: Int(workingSetGiB * Double(Self.GiB)),
            physicalMemoryBytes: Int(physicalGiB * Double(Self.GiB)),
            availableBytes: availableGiB.map { Int($0 * Double(Self.GiB)) }
        )
    }

    // MARK: - Formula

    @Test func m4Max64GBClampsToTheWholePool() throws {
        // 55.0 GiB working set, 48 GiB reclaimable: 48 − 1.025 − 10 − 0.06 − 5.5 = 31.4 GiB,
        // far above the 16.875 GiB pool, so the default is the whole pool.
        let d = try Derivation(host: Self.host(workingSetGiB: 55, physicalGiB: 64, availableGiB: 48), model: Self.qwen35B)
        #expect(d.poolBytes == Self.pool35B)
        #expect(d.floorBytes == Self.floor35B)
        #expect(d.kvReserveBytes == 10 * Self.GiB)
        let tenPercent = Int(5.5 * Double(Self.GiB))
        #expect(abs(d.marginBytes - tenPercent) <= 1, "10 % of 55 GiB outranks the 1 GiB minimum")
        #expect(d.plausibleAvailableBytes == 48 * Self.GiB)
        #expect(d.headroomBytes > d.poolBytes)
        #expect(d.derivedBytes == Self.pool35B)
        #expect(try d.admit(requestedGB: nil as Double?) == Self.pool35B)
        // The ceiling prices KV at the 4,096-token minimum against the working set:
        // 55 − 1.025 − 0.06 − 5.5 − 0.156 = 48.26 GiB, cut to the pool.
        #expect(d.ceilingBytes == Self.pool35B)
        // Today's fixed defaults all sit below the derived number and are honoured as ceilings.
        #expect(try d.admit(requestedGB: 8) == 8 * Self.GiB)
        #expect(try d.admit(requestedGB: 2) == 2 * Self.GiB)
        let pointFour = Int(0.4 * Double(Self.GiB))
        #expect(try d.admit(requestedGB: 0.4) == pointFour)
        // More than the pool can never be used: cut to the pool, not refused.
        #expect(try d.admit(requestedGB: 64) == Self.pool35B)
        let fullContext = d.contextTokensHoldable(besideCacheBytes: Self.pool35B)
        #expect(fullContext == 262_144)
    }

    @Test func thirtyTwoGBHostDerivesBelowThePool() throws {
        // 27.5 GiB working set, nothing else running: 27.5 − 1.025 − 10 − 0.06 − 2.75 = 13.67 GiB.
        let d = try Derivation(host: Self.host(workingSetGiB: 27.5, physicalGiB: 32, availableGiB: 30), model: Self.qwen35B)
        let workingSet = Int(27.5 * Double(Self.GiB))
        let kv = 10 * Self.GiB
        let fixed = 60 * Self.MiB
        let expected = workingSet - 1_101_009_920 - kv - fixed - d.marginBytes
        let tenPercent = Int(2.75 * Double(Self.GiB))
        #expect(abs(d.marginBytes - tenPercent) <= 1)
        #expect(d.headroomBytes == expected)
        #expect(d.derivedBytes == expected)
        #expect(d.derivedBytes > d.floorBytes && d.derivedBytes < d.poolBytes)
        // An explicit 16 GB is above the derived default but under the ceiling
        // (27.5 − 1.025 − 0.06 − 2.75 − 0.156 = 23.5 GiB): honoured, with less context headroom.
        #expect(try d.admit(requestedGB: 16) == 16 * Self.GiB)
        let headroomTokens = d.contextTokensHoldable(besideCacheBytes: 16 * Self.GiB)
        #expect(headroomTokens < 262_144)
        #expect(headroomTokens > 4_096)
        // 24 GB asks for more than the container holds; what would be allocated
        // is the pool, which fits under the ceiling, so it is cut, not refused.
        #expect(try d.admit(requestedGB: 24) == Self.pool35B)
    }

    @Test func sixteenGBHostFloorsTheDefaultAndRefusesAnOversizedCeiling() throws {
        // 10.67 GiB working set (two thirds of 16 GiB): KV at full context does not
        // fit at all, so the default is the floor, not a refusal.
        let d = try Derivation(host: Self.host(workingSetGiB: 10.67, physicalGiB: 16, availableGiB: 9), model: Self.qwen35B)
        #expect(d.headroomBytes < 0)
        #expect(d.derivedBytes == d.floorBytes)
        #expect(d.derivedBytes == Self.floor35B)
        // The ceiling: 10.67 − 1.025 − 0.06 − 1.067 − 0.156 = 8.36 GiB.
        let workingSet = Int(10.67 * Double(Self.GiB))
        let fixed = 60 * Self.MiB
        let minimumKV = 4_096 * Self.qwen35B.kvBytesPerToken
        let ceiling = workingSet - 1_101_009_920 - fixed - d.marginBytes - minimumKV
        let tenPercent = Int(1.067 * Double(Self.GiB))
        #expect(abs(d.marginBytes - tenPercent) <= 2_000, "10 % of the working set")
        #expect(d.ceilingBytes == ceiling)
        // The old CLI default of 8 GB is still honoured on this host...
        #expect(try d.admit(requestedGB: 8) == 8 * Self.GiB)
        // ...and 12 GB is refused naming the request, the ceiling and the shortfall.
        let twelve = 12 * Self.GiB
        let short = twelve - ceiling
        let refusal = Derivation.Error.budgetExceedsHost(requestedBytes: twelve, ceilingBytes: ceiling, shortBytes: short)
        #expect(throws: refusal) { try d.admit(requestedGB: 12) }
        let message = "\(refusal)"
        #expect(message.contains("12.00 GB"))
        #expect(message.contains(String(format: "%.2f GB", Double(ceiling) / Double(Self.GiB))))
        #expect(message.contains("short by"))
    }

    @Test func eightGBPhoneClassHostKeepsAWorkingCache() throws {
        // A 5 GiB allowance: resident 1.025 + floor 0.54 + 4,096 tokens of KV 0.16 + margin 1.0 fits;
        // the default is the floor and an explicit 2 GB (the app's historical value) is honoured.
        let d = try Derivation(host: Self.host(workingSetGiB: 5, physicalGiB: 8, availableGiB: 4), model: Self.qwen35B)
        #expect(d.marginBytes == Self.GiB, "the 1 GiB minimum margin outranks 10 % of 5 GiB")
        #expect(d.derivedBytes == Self.floor35B)
        #expect(try d.admit(requestedGB: 2) == 2 * Self.GiB)
        #expect(throws: Derivation.Error.self) { try d.admit(requestedGB: 4) }
    }

    @Test func hostThatCannotHoldTheModelIsRefusedAtDerivation() {
        // 2 GiB working set: 1.025 resident + 0.54 floor + 0.16 KV + 1.0 margin = 2.73 GiB needed.
        #expect(throws: Derivation.Error.self) {
            try Derivation(host: Self.host(workingSetGiB: 2, physicalGiB: 4, availableGiB: nil), model: Self.qwen35B)
        }
        do {
            _ = try Derivation(host: Self.host(workingSetGiB: 2, physicalGiB: 4, availableGiB: nil), model: Self.qwen35B)
            Issue.record("a 2 GiB working set admitted the 35B")
        } catch let error as Derivation.Error {
            guard case let .hostCannotHoldModel(needed, workingSet, short) = error else {
                Issue.record("wrong refusal: \(error)")
                return
            }
            #expect(workingSet == 2 * Self.GiB)
            let fixed = 60 * Self.MiB
            let margin = Self.GiB
            let minimumKV = 4_096 * Self.qwen35B.kvBytesPerToken
            let expectedNeeded = 1_101_009_920 + fixed + margin + minimumKV + Self.floor35B
            #expect(needed == expectedNeeded)
            #expect(short == needed - workingSet)
        } catch {
            Issue.record("wrong error type: \(error)")
        }
    }

    @Test func availableMemoryClampsTheDefaultOnlyWhenPlausible() throws {
        let busy = try Derivation(host: Self.host(workingSetGiB: 55, physicalGiB: 64, availableGiB: 20), model: Self.qwen35B)
        // 20 − 1.025 − 10 − 0.06 − 5.5 = 3.4 GiB: the reclaimable memory, not the working set, bounds the default...
        let offered = 20 * Self.GiB
        let kv = 10 * Self.GiB
        let fixed = 60 * Self.MiB
        let expected = offered - 1_101_009_920 - kv - fixed - busy.marginBytes
        #expect(busy.derivedBytes == expected)
        // ...while the ceiling for an explicit request is the GPU's working set.
        #expect(busy.ceilingBytes == Self.pool35B)

        // Under 2 % of physical memory the reading is implausible (macOS reports
        // single-digit free pages under heavy page-cache use) and is ignored.
        let implausible = try Derivation(host: Self.host(workingSetGiB: 55, physicalGiB: 64, availableGiB: 1), model: Self.qwen35B)
        #expect(implausible.plausibleAvailableBytes == nil)
        #expect(implausible.derivedBytes == Self.pool35B)

        let unmeasured = try Derivation(host: Self.host(workingSetGiB: 55, physicalGiB: 64, availableGiB: nil), model: Self.qwen35B)
        #expect(unmeasured.plausibleAvailableBytes == nil)
        #expect(unmeasured.derivedBytes == Self.pool35B)
        #expect(unmeasured.summary(budgetBytes: unmeasured.derivedBytes, requestedGB: nil as Double?).contains("unmeasured"))
    }

    @Test func floorIsOneTokensWorkingSetNeverBelowSixteenSlotsNeverAbovePool() throws {
        let host = Self.host(workingSetGiB: 55, physicalGiB: 64, availableGiB: nil)
        var small = Self.qwen35B
        small.expertFetchesPerToken = 4
        #expect(try Derivation(host: host, model: small).floorBytes == 16 * small.expertStride)
        var tiny = Self.qwen35B
        tiny.expertSlots = 8
        tiny.expertFetchesPerToken = 2
        #expect(try Derivation(host: host, model: tiny).floorBytes == 8 * tiny.expertStride)
        #expect(try Derivation(host: host, model: tiny).derivedBytes == 8 * tiny.expertStride)
    }

    @Test func invalidRequestsAndShapesAreRefusedByName() throws {
        let d = try Derivation(host: Self.host(workingSetGiB: 55, physicalGiB: 64, availableGiB: nil), model: Self.qwen35B)
        #expect(throws: Derivation.Error.invalidBudget("-1.0")) { try d.admit(requestedGB: -1) }
        #expect(throws: Derivation.Error.invalidBudget("nan")) { try d.admit(requestedGB: Double.nan) }
        #expect(throws: Derivation.Error.invalidBudget("inf")) { try d.admit(requestedGB: Double.infinity) }
        #expect(try d.admit(requestedGB: 0) == 0, "zero is well-formed; the cache's own minimum refuses it")

        var noStride = Self.qwen35B
        noStride.expertStride = 0
        #expect(throws: Derivation.Error.invalidModelShape("expert stride is 0")) {
            try Derivation(host: Self.host(workingSetGiB: 55, physicalGiB: 64, availableGiB: nil), model: noStride)
        }
        var noExperts = Self.qwen35B
        noExperts.expertSlots = 0
        #expect(throws: Derivation.Error.self) {
            try Derivation(host: Self.host(workingSetGiB: 55, physicalGiB: 64, availableGiB: nil), model: noExperts)
        }
    }

    @Test func summaryNamesEveryTerm() throws {
        let d = try Derivation(host: Self.host(workingSetGiB: 55, physicalGiB: 64, availableGiB: 48), model: Self.qwen35B)
        let derived = d.summary(budgetBytes: d.derivedBytes, requestedGB: nil as Double?)
        for needle in ["(derived)", "working set 55.00 GB", "available 48.00 GB", "resident dense 1.03 GB",
                       "KV 10.00 GB at 262144 tokens", "margin 5.50 GB", "clamped to the 16.88 GB pool",
                       "KV headroom beside it 262144 tokens"] {
            #expect(derived.contains(needle), "missing '\(needle)' in: \(derived)")
        }
        let explicit = d.summary(budgetBytes: try d.admit(requestedGB: 8), requestedGB: 8)
        #expect(explicit.contains("--cache-gb 8;"))
        #expect(explicit.contains("ceiling 16.88 GB"))
        #expect(explicit.contains("derived default 16.88 GB"))
        let cut = d.summary(budgetBytes: try d.admit(requestedGB: 64), requestedGB: 64)
        #expect(cut.contains("--cache-gb 64 cut to the pool"))
    }

    // MARK: - On the device

    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    static func repackedTiny() throws -> URL {
        let src = fixturesDir.appendingPathComponent("tiny-model-q4")
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("tiny-q4-budget-\(UUID().uuidString).qpack")
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        try repacker.repack()
        return out
    }

    /// Greedy continuation from a three-token prompt plus the per-step
    /// logits, so two runs can be compared byte for byte.
    static func greedy(_ model: QwenMetalModel, steps: Int = 8) throws -> (tokens: [Int], logits: [[Float]]) {
        let V = model.config.vocabSize
        let state = QwenCPUModel.DecodeState()
        var logits = try model.step([1, 5, 9], state: state)
        var tokens: [Int] = []
        var all: [[Float]] = [logits]
        for _ in 0..<steps {
            var best = 0
            for v in 1..<V where logits[v] > logits[best] { best = v }
            tokens.append(best)
            logits = try model.step([best], state: state)
            all.append(logits)
        }
        return (tokens, all)
    }

    /// The default on the tiny container: the host derives a budget, the
    /// pool (64 blobs) is far smaller than any Mac's headroom, so the cache
    /// is the whole pool and the derivation is exposed on the model.
    @Test func defaultBudgetOnTinyContainerIsTheWholePool() throws {
        let out = try Self.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let model = try QwenMetalModel(modelDir: out)
        let cache = try #require(model.expertCache)
        let derivation = try #require(model.cacheBudgetDerivation)
        #expect(model.requestedCacheBudgetGB == nil)
        #expect(derivation.model.expertSlots == 64)
        #expect(derivation.model.expertFetchesPerToken == 16)
        #expect(derivation.model.residentDenseBytes > 0, "the resident trunk is priced from the store")
        #expect(derivation.model.residentDenseBytes == model.store.residentBytes)
        let kvPerToken = 2 * 2 * (2 * 16) * 4
        #expect(derivation.model.kvBytesPerToken == kvPerToken, "2 full-attention layers, 2 KV heads × 16, f32 K and V")
        #expect(derivation.model.contextCapacity == model.contextCapacity)
        #expect(derivation.host.workingSetBytes > 0)
        #expect(derivation.derivedBytes == derivation.poolBytes)
        #expect(cache.budgetBytes == derivation.derivedBytes)
        #expect(cache.slotCount == 64)
    }

    /// An explicit ceiling on a real model: honoured when it fits, cut to the
    /// pool when it asks for more than the container holds.
    @Test func explicitCeilingOnTinyContainer() throws {
        let out = try Self.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let stride = try QpackExpertReader(containerDir: out).layout.expertStride
        let sixteenSlots = 16 * stride
        let sixteen = Double(sixteenSlots) / 1_073_741_824
        let model = try QwenMetalModel(modelDir: out, cacheBudgetGB: sixteen)
        let cache = try #require(model.expertCache)
        #expect(model.requestedCacheBudgetGB == sixteen)
        #expect(cache.slotCount == 16)
        #expect(cache.budgetBytes == 16 * stride)

        let oversized = try QwenMetalModel(modelDir: out, cacheBudgetGB: 0.05)
        #expect(try #require(oversized.expertCache).budgetBytes == 64 * stride, "cut to the pool")
        #expect(try #require(oversized.expertCache).slotCount == 64)
    }

    /// T13: the budget is a speed knob, never a correctness knob. The same
    /// greedy continuation at a 16-slot budget (one token's working set,
    /// eviction on every step) and at the whole 64-blob pool (no eviction)
    /// must produce byte-identical tokens and logits.
    @Test func greedyOutputIsByteIdenticalAcrossBudgets() throws {
        let out = try Self.repackedTiny()
        defer { try? FileManager.default.removeItem(at: out) }
        let stride = try QpackExpertReader(containerDir: out).layout.expertStride

        let sixteenSlotsGB = Double(16 * stride) / 1_073_741_824
        let evicting = try QwenMetalModel(modelDir: out, cacheBudgetGB: sixteenSlotsGB)
        let resident = try QwenMetalModel(modelDir: out)
        let evictingCache = try #require(evicting.expertCache)
        let residentCache = try #require(resident.expertCache)
        #expect(evictingCache.slotCount == 16)
        #expect(residentCache.slotCount == 64)

        let a = try Self.greedy(evicting)
        let b = try Self.greedy(resident)
        #expect(a.tokens.count == 8)
        #expect(a.tokens == b.tokens, "greedy tokens differ across budgets: \(a.tokens) vs \(b.tokens)")
        #expect(a.logits.count == b.logits.count)
        for (step, (x, y)) in zip(a.logits, b.logits).enumerated() {
            #expect(x == y, "step \(step): logits differ across budgets")
        }

        // The small budget really did evict: more misses than there are
        // blobs in the container, while the full pool missed each blob at
        // most once.
        #expect(residentCache.misses <= 64)
        #expect(evictingCache.misses > residentCache.misses,
                "16 slots: \(evictingCache.misses) misses; 64 slots: \(residentCache.misses) misses")
        #expect(evictingCache.allocatedBytes <= evictingCache.budgetBytes)
        #expect(residentCache.allocatedBytes <= residentCache.budgetBytes)
    }
}
