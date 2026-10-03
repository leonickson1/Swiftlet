import Foundation
import Testing
@testable import SwiftletCore

@Suite struct SwiftletSessionTests {
    private final class CleanupGate: @unchecked Sendable {
        private let lock = NSLock()
        private let releaseSemaphore = DispatchSemaphore(value: 0)
        private var _entered = false
        private var _completed = false
        private var calls = 0

        var entered: Bool { locked { _entered } }
        var completed: Bool { locked { _completed } }

        func run() {
            let shouldBlock = locked { () -> Bool in
                calls += 1
                if calls == 1 {
                    _entered = true
                    return true
                }
                return false
            }
            guard shouldBlock else { return }
            _ = releaseSemaphore.wait(timeout: .now() + .seconds(2))
            locked { _completed = true }
        }

        func release() {
            releaseSemaphore.signal()
        }

        private func locked<T>(_ body: () -> T) -> T {
            lock.lock()
            defer { lock.unlock() }
            return body()
        }
    }

    private final class LockedFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false

        var isSet: Bool {
            lock.lock()
            defer { lock.unlock() }
            return value
        }

        func set() {
            lock.lock()
            value = true
            lock.unlock()
        }
    }

    private final class SessionProbeModel: InferenceModel, @unchecked Sendable {
        enum ProbeError: Swift.Error { case timedOutWaitingForRelease }
        struct Call {
            let tokens: [Int]
            let stateID: ObjectIdentifier
        }

        let config: QwenConfig
        let modelDir: URL
        private let blockFirstCall: Bool
        private let cleanupGate: CleanupGate?
        private let lock = NSLock()
        private let releaseFirstCall = DispatchSemaphore(value: 0)
        private var _calls: [Call] = []
        private var _activeCalls = 0
        private var _maxActiveCalls = 0
        private var _firstCallEntered = false
        private var _secondStartedBeforeCleanup = false

        init(modelDir: URL, blockFirstCall: Bool, cleanupGate: CleanupGate? = nil) throws {
            self.modelDir = modelDir
            self.blockFirstCall = blockFirstCall
            self.cleanupGate = cleanupGate
            config = try QwenConfig(url: modelDir.appendingPathComponent("config.json"))
        }

        var calls: [Call] { locked { _calls } }
        var maxActiveCalls: Int { locked { _maxActiveCalls } }
        var firstCallEntered: Bool { locked { _firstCallEntered } }
        var secondStartedBeforeCleanup: Bool { locked { _secondStartedBeforeCleanup } }

        func releaseFirst() {
            releaseFirstCall.signal()
        }

        func step(_ tokens: [Int], state: QwenCPUModel.DecodeState) throws -> [Float] {
            try runStep(tokens: tokens, state: state, shouldCancel: { false })
        }

        func step(
            _ tokens: [Int],
            state: QwenCPUModel.DecodeState,
            shouldCancel: () -> Bool
        ) throws -> [Float] {
            try runStep(tokens: tokens, state: state, shouldCancel: shouldCancel)
        }

        private func runStep(
            tokens: [Int],
            state: QwenCPUModel.DecodeState,
            shouldCancel: () -> Bool
        ) throws -> [Float] {
            let ordinal = locked { () -> Int in
                _calls.append(Call(tokens: tokens, stateID: ObjectIdentifier(state)))
                _activeCalls += 1
                _maxActiveCalls = max(_maxActiveCalls, _activeCalls)
                if _calls.count == 1 { _firstCallEntered = true }
                return _calls.count
            }
            defer { locked { _activeCalls -= 1 } }

            if ordinal == 2, let cleanupGate, !cleanupGate.completed {
                locked { _secondStartedBeforeCleanup = true }
            }
            if blockFirstCall, ordinal == 1 {
                // Block like a long Metal/CPU step would, but observe
                // cancellation at the same cadence a real step does.
                let deadline = DispatchTime.now() + .seconds(2)
                while releaseFirstCall.wait(timeout: .now() + .milliseconds(5)) == .timedOut {
                    if shouldCancel() { throw GenerationInterruption.cancelled }
                    if DispatchTime.now() >= deadline {
                        throw ProbeError.timedOutWaitingForRelease
                    }
                }
            }
            if shouldCancel() { throw GenerationInterruption.cancelled }
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

    private static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    private func makeSession(
        model: SessionProbeModel,
        cleanupGate: CleanupGate? = nil
    ) -> SwiftletSession {
        SwiftletSession(
            testingModel: model,
            modelDir: model.modelDir,
            encodeText: { _ in [30] },
            decodeTokens: { tokens in
                String(String.UnicodeScalarView(tokens.compactMap { Unicode.Scalar($0) }))
            },
            renderMessages: { messages in
                // A transcript renders as the ids each turn was fed: the user
                // turn's token, then the assistant's reply bytes, then the next
                // user turn — so a continuation's render begins with exactly
                // what the state holds and the tail is the new turn alone.
                var ids: [Int] = []
                for message in messages {
                    switch (message["role"], message["content"]) {
                    case ("assistant", let reply?):
                        ids += reply.unicodeScalars.map { Int($0.value) }
                    case (_, "one"): ids.append(10)
                    case (_, "other"): ids.append(20)
                    default: ids.append(30)
                    }
                }
                return ids
            },
            generationCleanupHook: cleanupGate.map { gate in { gate.run() } }
        )
    }

    /// Issue #9: a token boundary that splits a multi-byte character decodes
    /// to a trailing U+FFFD, which must be held back, never emitted or stored
    /// as printed text (it poisons the prefix check and the EOS gate).
    @Test func trimIncompleteUTF8HoldsUnstableTail() {
        #expect(SwiftletSession.trimIncompleteUTF8("hello") == "hello")
        #expect(SwiftletSession.trimIncompleteUTF8("") == "")
        #expect(SwiftletSession.trimIncompleteUTF8("~10\u{FFFD}") == "~10")
        // A partial 4-byte sequence can render as several replacement chars.
        #expect(SwiftletSession.trimIncompleteUTF8("x\u{FFFD}\u{FFFD}\u{FFFD}") == "x")
        // Interior replacement chars are genuine content and stay.
        #expect(SwiftletSession.trimIncompleteUTF8("a\u{FFFD}b") == "a\u{FFFD}b")
        #expect(SwiftletSession.trimIncompleteUTF8("a\u{FFFD}b\u{FFFD}") == "a\u{FFFD}b")
        // Completed characters pass through untouched.
        #expect(SwiftletSession.trimIncompleteUTF8("~10⁵⁰⁰ vacua") == "~10⁵⁰⁰ vacua")
    }

    @Test func queuedCancellationCannotStepOrResetActiveConversation() async throws {
        let model = try SessionProbeModel(
            modelDir: Self.fixturesDir.appendingPathComponent("tiny-model"),
            blockFirstCall: true
        )
        let session = makeSession(model: model)
        var options = SwiftletSession.GenerationOptions.greedy
        options.minNew = 0

        let firstStream = session.streamChat(
            messages: [["role": "user", "content": "one"]],
            maxNew: 1,
            options: options
        )
        let first = Task {
            var output = ""
            for try await delta in firstStream { output += delta }
            return output
        }
        let deadline = Date().addingTimeInterval(1)
        while !model.firstCallEntered, Date() < deadline {
            try await Task<Never, Never>.sleep(nanoseconds: 1_000_000)
        }
        #expect(model.firstCallEntered)

        let queuedCancellation = GenerationCancellation()
        let queuedStream = session.streamChat(
            messages: [["role": "user", "content": "other"]],
            maxNew: 1,
            options: options,
            cancellation: queuedCancellation
        )
        let queued = Task {
            var output = ""
            for try await delta in queuedStream { output += delta }
            return output
        }
        queuedCancellation.cancel()
        model.releaseFirst()

        let firstOutput = try await first.value
        let queuedOutput = try await queued.value
        #expect(firstOutput == "A")
        #expect(queuedOutput == "")
        #expect(model.calls.map(\.tokens) == [[10], [65]])
        #expect(model.maxActiveCalls == 1)

        // The cancelled request was queued, so it must not reset the state
        // committed by the first request. This transcript renders to the ids
        // the state was fed plus one new token, so the prefix is reused and
        // only that token is fed to the same DecodeState identity.
        let continuationStream = session.streamChat(
            messages: [
                ["role": "user", "content": "one"],
                ["role": "assistant", "content": "A"],
                ["role": "user", "content": "next"],
            ],
            maxNew: 0,
            options: options
        )
        for try await _ in continuationStream {}

        let calls = model.calls
        #expect(calls.map(\.tokens) == [[10], [65], [30]])
        #expect(Set(calls.map(\.stateID)).count == 1)
        #expect(model.maxActiveCalls == 1)
    }

    /// A "new chat" issued mid-reply must interrupt the reply, not wait for it.
    /// The probe blocks its first step for 2 s unless cancelled; the reset must
    /// return well inside that window, and the following request must be a
    /// fresh prompt rather than a continuation of the interrupted one.
    @Test func resetConversationCancelsActiveGenerationInsteadOfWaiting() async throws {
        let model = try SessionProbeModel(
            modelDir: Self.fixturesDir.appendingPathComponent("tiny-model"),
            blockFirstCall: true
        )
        let session = makeSession(model: model)
        var options = SwiftletSession.GenerationOptions.greedy
        options.minNew = 0

        let stream = session.streamChat(
            messages: [["role": "user", "content": "one"]],
            maxNew: 4,
            options: options
        )
        let consumer = Task { () -> Swift.Error? in
            do {
                for try await _ in stream {}
                return nil
            } catch {
                return error
            }
        }
        let deadline = Date().addingTimeInterval(1)
        while !model.firstCallEntered, Date() < deadline {
            try await Task<Never, Never>.sleep(nanoseconds: 1_000_000)
        }
        #expect(model.firstCallEntered)

        let resetStarted = Date()
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            DispatchQueue.global().async {
                session.resetConversation()
                done.resume()
            }
        }
        let resetSeconds = Date().timeIntervalSince(resetStarted)
        #expect(resetSeconds < 1.0, "reset waited \(resetSeconds)s for the interrupted reply")

        // A cooperative cancellation ends the stream cleanly; the session reports
        // it through metrics rather than by throwing to the consumer.
        let failure = await consumer.value
        #expect(failure == nil)
        #expect(session.lastMetrics.finishReason == .cancelled)
        #expect(model.calls.map(\.tokens) == [[10]])

        // The reset landed: the same first message renders as a fresh prompt.
        let next = session.streamChat(
            messages: [["role": "user", "content": "one"]],
            maxNew: 1,
            options: options
        )
        var nextOutput = ""
        for try await delta in next { nextOutput += delta }
        #expect(nextOutput == "A")
        #expect(model.calls.map(\.tokens) == [[10], [10], [65]])
        #expect(model.maxActiveCalls == 1)
    }

    @Test func concurrentStreamChatsCannotOverlapOrReplaceActiveState() async throws {
        let model = try SessionProbeModel(
            modelDir: Self.fixturesDir.appendingPathComponent("tiny-model"),
            blockFirstCall: true
        )
        let session = makeSession(model: model)
        var options = SwiftletSession.GenerationOptions.greedy
        options.minNew = 0

        let firstStream = session.streamChat(
            messages: [["role": "user", "content": "one"]],
            maxNew: 1,
            options: options
        )
        let first = Task {
            var output = ""
            for try await delta in firstStream { output += delta }
            return output
        }
        let firstDeadline = Date().addingTimeInterval(1)
        while !model.firstCallEntered, Date() < firstDeadline {
            try await Task<Never, Never>.sleep(nanoseconds: 1_000_000)
        }
        #expect(model.firstCallEntered)

        // This prompt is deliberately unrelated, so it must reset state —
        // but only after the first request has fully finished using its own
        // state. The blocked first step makes any overlap observable.
        let secondStream = session.streamChat(
            messages: [["role": "user", "content": "other"]],
            maxNew: 1,
            options: options
        )
        let second = Task {
            var output = ""
            for try await delta in secondStream { output += delta }
            return output
        }
        try await Task<Never, Never>.sleep(nanoseconds: 20_000_000)
        #expect(model.calls.count == 1)

        model.releaseFirst()
        let firstOutput = try await first.value
        let secondOutput = try await second.value
        #expect(firstOutput == "A")
        #expect(secondOutput == "A")

        let calls = model.calls
        #expect(calls.map(\.tokens) == [[10], [65], [20], [65]])
        try #require(calls.count == 4)
        #expect(calls[0].stateID == calls[1].stateID)
        #expect(calls[2].stateID == calls[3].stateID)
        #expect(calls[0].stateID != calls[2].stateID)
        #expect(model.maxActiveCalls == 1)
    }

    @Test func cleanupCompletesBeforeContinuationOrNextRequestStarts() async throws {
        let cleanupGate = CleanupGate()
        let model = try SessionProbeModel(
            modelDir: Self.fixturesDir.appendingPathComponent("tiny-model"),
            blockFirstCall: false,
            cleanupGate: cleanupGate
        )
        let session = makeSession(model: model, cleanupGate: cleanupGate)
        let firstStarted = LockedFlag()
        let firstFinished = LockedFlag()

        let firstStream = session.streamChat(
            messages: [["role": "user", "content": "one"]], maxNew: 0
        )
        let first = Task {
            firstStarted.set()
            for try await _ in firstStream {}
            firstFinished.set()
        }
        let deadline = Date().addingTimeInterval(1)
        while (!firstStarted.isSet || !cleanupGate.entered), Date() < deadline {
            try await Task<Never, Never>.sleep(nanoseconds: 1_000_000)
        }
        #expect(firstStarted.isSet)
        #expect(cleanupGate.entered)

        let secondStream = session.streamChat(
            messages: [["role": "user", "content": "other"]], maxNew: 0
        )
        let second = Task {
            for try await _ in secondStream {}
        }
        try await Task<Never, Never>.sleep(nanoseconds: 20_000_000)

        #expect(!firstFinished.isSet)
        #expect(model.calls.count == 1)
        cleanupGate.release()
        try await first.value
        try await second.value

        #expect(firstFinished.isSet)
        #expect(cleanupGate.completed)
        #expect(model.calls.count == 2)
        #expect(!model.secondStartedBeforeCleanup)
        #expect(model.maxActiveCalls == 1)
    }
}
