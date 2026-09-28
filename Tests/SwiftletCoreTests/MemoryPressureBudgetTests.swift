import Foundation
import Testing
@testable import SwiftletCore

/// OS memory pressure shrinks the expert cache and a return to normal
/// restores the configured budget. Before this `handleMemoryPressure()` had
/// no caller in the repository (nothing created a `DispatchSource` for it),
/// the shrink target was a literal 0.4 GB, and no path ever rebuilt the cache
/// at the `cacheBudgetGB` the session was configured with: one warning left
/// a server at 0.4 GB for the rest of its life. The session now decides a
/// target per level (`.warning`/`.critical` -> the valve, `.normal` -> the
/// configured budget), rebuilds the cache only when the target differs from
/// what the cache runs at, and `swiftlet-server` registers the source.
@Suite struct MemoryPressureBudgetTests {
    static let fixturesDir = MetalModelTests.fixturesDir

    /// A model with a resizable expert-cache budget and a first step that can
    /// be held open, so a pressure event can land mid-generation.
    private final class BudgetProbeModel: InferenceModel, ExpertCacheResizing, @unchecked Sendable {
        enum ProbeError: Swift.Error { case timedOutWaitingForRelease }

        let config: QwenConfig
        let modelDir: URL
        private let blockFirstCall: Bool
        private let lock = NSLock()
        private let releaseFirstCall = DispatchSemaphore(value: 0)
        /// Kept in GB, as asked for: a byte round trip (`Int(0.4 GiB) / GiB`)
        /// is 0.3999…, which is not the contract under test.
        private var _budgetGB: Double
        private var _resizes: [Double] = []
        private var _calls = 0
        private var _firstCallEntered = false
        private var _resizeDuringStep = false
        private var _activeSteps = 0

        init(modelDir: URL, budgetGB: Double, blockFirstCall: Bool = false) throws {
            self.modelDir = modelDir
            self.blockFirstCall = blockFirstCall
            config = try QwenConfig(url: modelDir.appendingPathComponent("config.json"))
            _budgetGB = budgetGB
        }

        var resizes: [Double] { locked { _resizes } }
        var budgetGB: Double { locked { _budgetGB } }
        var firstCallEntered: Bool { locked { _firstCallEntered } }
        var resizeDuringStep: Bool { locked { _resizeDuringStep } }

        func releaseFirst() { releaseFirstCall.signal() }

        // ExpertCacheResizing
        var expertCacheBudgetBytes: Int? { locked { Int(_budgetGB * 1_073_741_824) } }
        func resizeExpertCache(toGB gb: Double) {
            locked {
                if _activeSteps > 0 { _resizeDuringStep = true }
                _resizes.append(gb)
                _budgetGB = gb
            }
        }

        func step(_ tokens: [Int], state: QwenCPUModel.DecodeState) throws -> [Float] {
            try step(tokens, state: state, shouldCancel: { false })
        }

        func step(
            _ tokens: [Int], state: QwenCPUModel.DecodeState, shouldCancel: () -> Bool
        ) throws -> [Float] {
            let ordinal = locked { () -> Int in
                _calls += 1
                _activeSteps += 1
                if _calls == 1 { _firstCallEntered = true }
                return _calls
            }
            defer { locked { _activeSteps -= 1 } }
            if blockFirstCall, ordinal == 1 {
                let deadline = DispatchTime.now() + .seconds(2)
                while releaseFirstCall.wait(timeout: .now() + .milliseconds(5)) == .timedOut {
                    if shouldCancel() { throw GenerationInterruption.cancelled }
                    if DispatchTime.now() >= deadline { throw ProbeError.timedOutWaitingForRelease }
                }
            }
            state.position += tokens.count
            var logits = [Float](repeating: -.infinity, count: config.vocabSize)
            logits[65] = 1
            return logits
        }

        private func locked<T>(_ body: () -> T) -> T {
            lock.lock()
            defer { lock.unlock() }
            return body()
        }
    }

    private static func makeSession(model: BudgetProbeModel, cacheBudgetGB: Double) -> SwiftletSession {
        SwiftletSession(
            testingModel: model,
            modelDir: model.modelDir,
            encodeText: { _ in [30] },
            decodeTokens: { tokens in
                String(String.UnicodeScalarView(tokens.compactMap { Unicode.Scalar($0) }))
            },
            renderMessages: { _ in [10] },
            cacheBudgetGB: cacheBudgetGB
        )
    }

    private static func probe(budgetGB: Double, blockFirstCall: Bool = false) throws -> BudgetProbeModel {
        try BudgetProbeModel(
            modelDir: fixturesDir.appendingPathComponent("tiny-model"),
            budgetGB: budgetGB, blockFirstCall: blockFirstCall)
    }

    // MARK: idle

    /// The contract: a warning shrinks to the valve, normal restores exactly
    /// the configured budget, and the session reports what the cache runs at.
    @Test func warningShrinksToTheValveAndNormalRestoresTheConfiguredBudget() throws {
        let model = try Self.probe(budgetGB: 2)
        let session = Self.makeSession(model: model, cacheBudgetGB: 2)
        #expect(session.cacheBudgetGB == 2)
        #expect(session.currentCacheBudgetGB == 2)

        session.handleMemoryPressure(.warning)
        #expect(model.resizes == [SwiftletSession.pressureShrinkGB])
        #expect(model.budgetGB == SwiftletSession.pressureShrinkGB)
        #expect(session.currentCacheBudgetGB == SwiftletSession.pressureShrinkGB)

        session.handleMemoryPressure(.normal)
        #expect(model.resizes == [SwiftletSession.pressureShrinkGB, 2])
        #expect(model.budgetGB == 2, "normal must restore the configured budget, not leave the valve in place")
        #expect(session.currentCacheBudgetGB == 2)
    }

    /// The zero-argument entry point the iOS host already calls keeps its
    /// meaning: a warning.
    @Test func zeroArgumentEntryPointIsAWarning() throws {
        let model = try Self.probe(budgetGB: 2)
        let session = Self.makeSession(model: model, cacheBudgetGB: 2)
        session.handleMemoryPressure()
        #expect(model.resizes == [SwiftletSession.pressureShrinkGB])
    }

    /// Rebuilding the cache drops every resident expert, so a level that does
    /// not change the target must not rebuild: normal with nothing to restore,
    /// a second warning, critical after warning.
    @Test func aLevelThatDoesNotChangeTheTargetRebuildsNothing() throws {
        let model = try Self.probe(budgetGB: 2)
        let session = Self.makeSession(model: model, cacheBudgetGB: 2)

        session.handleMemoryPressure(.normal)
        #expect(model.resizes.isEmpty, "normal with no preceding shrink must not touch the cache")

        session.handleMemoryPressure(.warning)
        session.handleMemoryPressure(.warning)
        session.handleMemoryPressure(.critical)
        #expect(model.resizes == [SwiftletSession.pressureShrinkGB], "repeated pressure must not rebuild the already-shrunk cache")

        session.handleMemoryPressure(.normal)
        session.handleMemoryPressure(.normal)
        #expect(model.resizes == [SwiftletSession.pressureShrinkGB, 2])
    }

    /// Critical shrinks like warning does today.
    @Test func criticalShrinksToTheValve() throws {
        let model = try Self.probe(budgetGB: 4)
        let session = Self.makeSession(model: model, cacheBudgetGB: 4)
        session.handleMemoryPressure(.critical)
        #expect(model.resizes == [SwiftletSession.pressureShrinkGB])
        session.handleMemoryPressure(.normal)
        #expect(model.resizes == [SwiftletSession.pressureShrinkGB, 4])
    }

    /// A budget configured below the valve is never grown by a warning: the
    /// valve is a ceiling under pressure, not a floor.
    @Test func aConfiguredBudgetBelowTheValveIsNeverGrown() throws {
        let model = try Self.probe(budgetGB: 0.25)
        let session = Self.makeSession(model: model, cacheBudgetGB: 0.25)
        session.handleMemoryPressure(.warning)
        session.handleMemoryPressure(.normal)
        #expect(model.resizes.isEmpty)
        #expect(model.budgetGB == 0.25)
        #expect(session.currentCacheBudgetGB == 0.25)
    }

    // MARK: mid-generation

    /// Pressure during a generation is applied between tokens, never while a
    /// step is running (the race garbles expert reads), and the restore is
    /// deferred the same way.
    @Test func pressureDuringGenerationIsAppliedBetweenTokens() async throws {
        let model = try Self.probe(budgetGB: 2, blockFirstCall: true)
        let session = Self.makeSession(model: model, cacheBudgetGB: 2)
        var options = SwiftletSession.GenerationOptions.greedy
        options.minNew = 0

        let stream = session.streamChat(messages: [["role": "user", "content": "one"]], maxNew: 1, options: options)
        let reply = Task {
            var output = ""
            for try await delta in stream { output += delta }
            return output
        }
        let deadline = Date().addingTimeInterval(1)
        while !model.firstCallEntered, Date() < deadline {
            try await Task<Never, Never>.sleep(nanoseconds: 1_000_000)
        }
        #expect(model.firstCallEntered)

        session.handleMemoryPressure(.warning)
        #expect(model.resizes.isEmpty, "the shrink must wait for the step to finish")
        #expect(session.currentCacheBudgetGB == 2)

        model.releaseFirst()
        _ = try await reply.value
        #expect(model.resizes == [SwiftletSession.pressureShrinkGB])
        #expect(!model.resizeDuringStep)
        #expect(session.currentCacheBudgetGB == SwiftletSession.pressureShrinkGB)

        session.handleMemoryPressure(.normal)
        #expect(model.resizes == [SwiftletSession.pressureShrinkGB, 2])
        #expect(session.currentCacheBudgetGB == 2)
    }

    /// A return to normal that lands while a warning's shrink is still queued
    /// cancels the queued shrink: the cache the decode thread would have torn
    /// down between tokens is the one it should keep.
    @Test func normalDuringGenerationCancelsAQueuedShrink() async throws {
        let model = try Self.probe(budgetGB: 2, blockFirstCall: true)
        let session = Self.makeSession(model: model, cacheBudgetGB: 2)
        var options = SwiftletSession.GenerationOptions.greedy
        options.minNew = 0

        let stream = session.streamChat(messages: [["role": "user", "content": "one"]], maxNew: 1, options: options)
        let reply = Task {
            var output = ""
            for try await delta in stream { output += delta }
            return output
        }
        let deadline = Date().addingTimeInterval(1)
        while !model.firstCallEntered, Date() < deadline {
            try await Task<Never, Never>.sleep(nanoseconds: 1_000_000)
        }
        #expect(model.firstCallEntered)

        session.handleMemoryPressure(.warning)
        session.handleMemoryPressure(.normal)
        model.releaseFirst()
        _ = try await reply.value
        #expect(model.resizes.isEmpty, "a cancelled shrink must not rebuild the cache at either budget")
        #expect(session.currentCacheBudgetGB == 2)
    }

    // MARK: the OS source

    /// The monitor the server installs listens for all three levels; without
    /// `.normal` the budget could never be restored.
    @Test func monitorRegistersWarningCriticalAndNormal() throws {
        let model = try Self.probe(budgetGB: 2)
        let session = Self.makeSession(model: model, cacheBudgetGB: 2)
        let monitor = session.makeMemoryPressureMonitor()
        defer { monitor.cancel() }
        #expect(monitor.mask.contains(.warning))
        #expect(monitor.mask.contains(.critical))
        #expect(monitor.mask.contains(.normal))
        #expect(!monitor.isCancelled)
        #expect(model.resizes.isEmpty, "registering the source must not touch the cache")
    }

    /// A dispatch event can carry several bits; the most severe one decides.
    @Test func levelFromEventPrefersTheMostSevereBit() {
        #expect(MemoryPressureLevel(event: .warning) == .warning)
        #expect(MemoryPressureLevel(event: .critical) == .critical)
        #expect(MemoryPressureLevel(event: .normal) == .normal)
        #expect(MemoryPressureLevel(event: [.warning, .critical]) == .critical)
        #expect(MemoryPressureLevel(event: [.normal, .warning]) == .warning)
        #expect(MemoryPressureLevel(event: []) == nil)
    }
}
