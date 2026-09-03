import Foundation
import Testing
@testable import SwiftletCore

/// Context admission at the session boundary, through the dependency-injected
/// `SwiftletSession` initializer: a probe model with a small effective
/// capacity, fixed token rendering, and greedy sampling.
@Suite struct ContextAdmissionSessionTests {
    private static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    /// Records every step and advances the position like the real models.
    /// The next token is fixed by position (A, B, C, A, ...) with a wide
    /// margin, so replies are predictable and never trip the repetition
    /// penalties or the n-gram guard within one short turn.
    private final class CapacityProbeModel: InferenceModel, @unchecked Sendable {
        let config: QwenConfig
        let modelDir: URL
        let contextCapacity: Int
        private let lock = NSLock()
        private var _calls: [(tokens: [Int], stateID: ObjectIdentifier)] = []

        init(modelDir: URL, contextCapacity: Int) throws {
            self.modelDir = modelDir
            self.contextCapacity = contextCapacity
            config = try QwenConfig(url: modelDir.appendingPathComponent("config.json"))
        }

        var calls: [[Int]] {
            lock.lock()
            defer { lock.unlock() }
            return _calls.map(\.tokens)
        }

        var stateIDs: Set<ObjectIdentifier> {
            lock.lock()
            defer { lock.unlock() }
            return Set(_calls.map(\.stateID))
        }

        func step(_ tokens: [Int], state: QwenCPUModel.DecodeState) throws -> [Float] {
            lock.lock()
            _calls.append((tokens, ObjectIdentifier(state)))
            lock.unlock()
            state.position += tokens.count
            var logits = [Float](repeating: 0, count: config.vocabSize)
            logits[65 + state.position % 3] = 10
            return logits
        }
    }

    private static var greedy: SwiftletSession.GenerationOptions {
        var options = SwiftletSession.GenerationOptions.greedy
        options.minNew = 0
        return options
    }

    private func makeSession(model: CapacityProbeModel) -> SwiftletSession {
        SwiftletSession(
            testingModel: model,
            modelDir: model.modelDir,
            // Continuation turns: "short" renders to one token, anything else
            // to four, so a caller can choose which side of the window it is on.
            encodeText: { text in text.contains("short") ? [30] : [30, 31, 32, 33] },
            decodeTokens: { tokens in
                String(String.UnicodeScalarView(tokens.compactMap { Unicode.Scalar($0) }))
            },
            renderMessages: { _ in [10, 11, 12] }
        )
    }

    private func collect(_ stream: AsyncThrowingStream<String, Swift.Error>) async throws -> String {
        var output = ""
        for try await delta in stream { output += delta }
        return output
    }

    private func probe(capacity: Int) throws -> CapacityProbeModel {
        try CapacityProbeModel(
            modelDir: Self.fixturesDir.appendingPathComponent("tiny-model"),
            contextCapacity: capacity
        )
    }

    @Test func rejectedContinuationKeepsTheConversation() async throws {
        let model = try probe(capacity: 8)
        let session = makeSession(model: model)

        // 3 prompt tokens + 2 generated: position 5 of 8.
        let first = try await collect(session.streamChat(
            messages: [["role": "user", "content": "one"]], maxNew: 2, options: Self.greedy
        ))
        #expect(first == "AB")
        #expect(session.lastMetrics.finishReason == .length)
        #expect(model.calls == [[10, 11, 12], [65], [66]])

        // 5 processed + 4 incoming exceeds 8: rejected before any model call,
        // and the cached conversation is left intact.
        let history = [
            ["role": "user", "content": "one"],
            ["role": "assistant", "content": "AB"],
        ]
        await #expect(throws: ContextWindowError.capacityExceeded(processed: 5, incoming: 4, maximum: 8)) {
            _ = try await self.collect(session.streamChat(
                messages: history + [["role": "user", "content": "long"]],
                maxNew: 2, options: Self.greedy
            ))
        }
        #expect(model.calls.count == 3)

        // A shorter retry continues the same decode state: one suffix token,
        // then two generated, filling the window exactly.
        let third = try await collect(session.streamChat(
            messages: history + [["role": "user", "content": "short"]],
            maxNew: 2, options: Self.greedy
        ))
        #expect(third == "AB")
        #expect(model.calls == [[10, 11, 12], [65], [66], [30], [65], [66]])
        #expect(model.stateIDs.count == 1)

        // Full window: any continuation is rejected; a fresh conversation
        // starts from a new state.
        let fullHistory = history + [
            ["role": "user", "content": "short"],
            ["role": "assistant", "content": "AB"],
        ]
        await #expect(throws: ContextWindowError.capacityExceeded(processed: 8, incoming: 1, maximum: 8)) {
            _ = try await self.collect(session.streamChat(
                messages: fullHistory + [["role": "user", "content": "short"]],
                maxNew: 1, options: Self.greedy
            ))
        }
        #expect(model.calls.count == 6)
        let fresh = try await collect(session.streamChat(
            messages: [["role": "user", "content": "two"]], maxNew: 1, options: Self.greedy
        ))
        #expect(fresh == "A")
        #expect(model.calls.count == 8)
        #expect(model.stateIDs.count == 2)
    }

    @Test func clampsOutputToRemainingCapacityAndReportsLength() async throws {
        let model = try probe(capacity: 8)
        let session = makeSession(model: model)

        let output = try await collect(session.streamChat(
            messages: [["role": "user", "content": "one"]], maxNew: 10, options: Self.greedy
        ))
        #expect(output == "ABCAB")
        #expect(session.lastMetrics.generatedTokens == 5)
        #expect(session.lastMetrics.finishReason == .length)
        #expect(model.calls.count == 6)
    }

    @Test func promptWithNoGenerationRoomIsRejectedBeforeAnyStep() async throws {
        let model = try probe(capacity: 3)
        let session = makeSession(model: model)

        await #expect(throws: ContextWindowError.noGenerationCapacity(processed: 0, incoming: 3, maximum: 3)) {
            _ = try await self.collect(session.streamChat(
                messages: [["role": "user", "content": "one"]], maxNew: 1, options: Self.greedy
            ))
        }
        #expect(model.calls.isEmpty)

        // Prefill-only is admitted up to the full window.
        let output = try await collect(session.streamChat(
            messages: [["role": "user", "content": "one"]], maxNew: 0, options: Self.greedy
        ))
        #expect(output == "")
        #expect(session.lastMetrics.generatedTokens == 0)
        #expect(model.calls == [[10, 11, 12]])
    }
}
