import Foundation
import Testing
@testable import SwiftletCore

/// Context admission through the production `SwiftletSession` initializer:
/// the tiny model plus its byte-level tokenizer fixture (one token per
/// character, `max_position_embeddings` 512), so prompt sizes are exact.
@Suite struct ContextAdmissionTokenizerTests {
    private static let modelDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")
        .appendingPathComponent("tiny-model")

    private static var greedy: SwiftletSession.GenerationOptions {
        var options = SwiftletSession.GenerationOptions.greedy
        options.minNew = 0
        return options
    }

    private func collect(_ stream: AsyncThrowingStream<String, Swift.Error>) async throws -> String {
        var output = ""
        for try await delta in stream { output += delta }
        return output
    }

    /// `<|im_start|>user\n` + content + `<|im_end|>\n<|im_start|>assistant\n`
    /// is 19 markup tokens around the content with this tokenizer.
    @Test func oversizedPromptIsRejectedAndConversationSurvives() async throws {
        let session = try await SwiftletSession(modelDir: Self.modelDir)

        await #expect(throws: ContextWindowError.capacityExceeded(processed: 0, incoming: 619, maximum: 512)) {
            _ = try await self.collect(session.streamChat(
                messages: [["role": "user", "content": String(repeating: "a", count: 600)]],
                maxNew: 1, options: Self.greedy
            ))
        }

        // A prompt that exactly fills the window is refused generation room
        // but admitted for prefill only.
        let filling = String(repeating: "a", count: 512 - 19)
        await #expect(throws: ContextWindowError.noGenerationCapacity(processed: 0, incoming: 512, maximum: 512)) {
            _ = try await self.collect(session.streamChat(
                messages: [["role": "user", "content": filling]], maxNew: 1, options: Self.greedy
            ))
        }
        let prefillOnly = try await collect(session.streamChat(
            messages: [["role": "user", "content": filling]], maxNew: 0, options: Self.greedy
        ))
        #expect(prefillOnly == "")
        #expect(session.lastMetrics.promptTokens == 512)
        #expect(session.lastMetrics.generatedTokens == 0)

        // A normal turn, then an oversized continuation, then a short one:
        // the short one continues the cached state (only its own turn is
        // prefilled) instead of re-rendering the whole conversation.
        let reply = try await collect(session.streamChat(
            messages: [["role": "user", "content": "hi"]], maxNew: 3, options: Self.greedy
        ))
        #expect(session.lastMetrics.promptTokens == 21)
        #expect(session.lastMetrics.generatedTokens == 3)
        #expect(session.lastMetrics.finishReason == .length)
        let history = [
            ["role": "user", "content": "hi"],
            ["role": "assistant", "content": reply],
        ]
        await #expect(throws: ContextWindowError.self) {
            _ = try await self.collect(session.streamChat(
                messages: history + [["role": "user", "content": String(repeating: "a", count: 600)]],
                maxNew: 1, options: Self.greedy
            ))
        }
        _ = try await collect(session.streamChat(
            messages: history + [["role": "user", "content": "ok"]], maxNew: 1, options: Self.greedy
        ))
        // Continuation suffix: <|im_end|>\n<|im_start|>user\nok<|im_end|>\n<|im_start|>assistant\n
        #expect(session.lastMetrics.promptTokens == 23)
        #expect(session.lastMetrics.generatedTokens == 1)
    }
}
