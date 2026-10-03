import Foundation
import Testing
@testable import SwiftletCore

/// Port of colibri's `PrefixReuseContract` (`c/tests/test_inkling_prefix_serve.py`
/// over `c/kv_prefix.h`): reusing a previous turn's state must change nothing
/// but the time. Two turns on one session, the second reusing the first's
/// state, must produce byte-identical output to the second turn alone on a
/// cold session; every rejection path — a changed earlier message, the
/// off-switch, a thinking-family re-render — must equal cold as well.
///
/// The model here answers from its whole fed history, so a state that belongs
/// to a different conversation produces different bytes: the contract can
/// fail, and did against the string-keyed reuse this replaces (a warm
/// thinking-family turn held an empty think block that a cold prefill of the
/// same transcript does not).
@Suite struct PrefixReuseContract {
    private static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    // Fake tokenizer: every byte is its own id; four small ids are the
    // template's special tokens.
    private static let thinkOpen = 1      // "<think>\n"
    private static let thinkClose = 2     // "\n</think>\n\n"
    private static let imStart = 3        // "<|im_start|>"
    private static let imEnd = 4          // "<|im_end|>"

    private static func bytes(_ text: String) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }

    private static func encode(_ text: String) -> [Int] {
        switch text {
        case "<think>\n": return [thinkOpen]
        case "\n</think>\n\n": return [thinkClose]
        default: return bytes(text)
        }
    }

    private static func decode(_ ids: [Int]) -> String {
        String(String.UnicodeScalarView(ids.compactMap { $0 >= 32 ? Unicode.Scalar($0) : nil }))
    }

    /// The chat template's shape: `<|im_start|>role\ncontent<|im_end|>\n` per
    /// message, then the generation prompt. A thinking-family template ends it
    /// with `<think>\n` (the session closes the block itself) and, like
    /// Qwen3's, renders a PAST assistant turn without any think block — which
    /// is exactly what makes a re-render diverge from the fed ids.
    private static func render(_ messages: [[String: String]], think: Bool) -> [Int] {
        var ids: [Int] = []
        for message in messages {
            ids += [imStart] + bytes(message["role"] ?? "") + [10]
            ids += bytes(message["content"] ?? "")
            ids += [imEnd, 10]
        }
        ids += [imStart] + bytes("assistant") + [10]
        if think { ids.append(thinkOpen) }
        return ids
    }

    /// Answers from its whole fed history: the next token is a function of
    /// every id the state has consumed, so two states built from different
    /// ids rank the vocabulary differently. Positions advance like the real
    /// models; a state at position 0 is fresh and starts a new history.
    private final class HistoryModel: InferenceModel, @unchecked Sendable {
        let config: QwenConfig
        let modelDir: URL
        private let lock = NSLock()
        private var histories: [ObjectIdentifier: [Int]] = [:]
        private var _calls: [[Int]] = []
        /// Optional fault: on this call ordinal, advance the state and then
        /// report cancellation — the shape of a Metal step interrupted between
        /// command buffers.
        var cancelOnCall: (ordinal: Int, cancellation: GenerationCancellation)?

        init(modelDir: URL) throws {
            self.modelDir = modelDir
            config = try QwenConfig(url: modelDir.appendingPathComponent("config.json"))
        }

        var calls: [[Int]] { lock.lock(); defer { lock.unlock() }; return _calls }

        func step(_ tokens: [Int], state: QwenCPUModel.DecodeState) throws -> [Float] {
            lock.lock()
            defer { lock.unlock() }
            _calls.append(tokens)
            let key = ObjectIdentifier(state)
            var history = state.position == 0 ? [] : (histories[key] ?? [])
            history += tokens
            histories[key] = history
            state.position += tokens.count
            if let fault = cancelOnCall, fault.ordinal == _calls.count {
                fault.cancellation.cancel()
                throw GenerationInterruption.cancelled
            }
            var h = 7
            for id in history { h = (h &* 31 &+ id &+ 1) % 1_000_003 }
            // Rank A..Z by distance from the history's pick, so a banned token
            // falls to the next in a deterministic order rather than to id 0.
            var logits = [Float](repeating: -.infinity, count: config.vocabSize)
            for v in 65...90 {
                let rank = (v - 65 - h % 26 + 26) % 26
                logits[v] = Float(26 - rank)
            }
            return logits
        }
    }

    private struct Harness {
        let model: HistoryModel
        let session: SwiftletSession
    }

    private static func makeHarness(think: Bool) throws -> Harness {
        let model = try HistoryModel(modelDir: fixturesDir.appendingPathComponent("tiny-model"))
        let session = SwiftletSession(
            testingModel: model,
            modelDir: model.modelDir,
            encodeText: { encode($0) },
            decodeTokens: { decode($0) },
            renderMessages: { render($0, think: think) },
            usesThinkPrompt: think
        )
        return Harness(model: model, session: session)
    }

    private static var greedy: SwiftletSession.GenerationOptions {
        var options = SwiftletSession.GenerationOptions.greedy
        options.minNew = 0
        return options
    }

    private static let turns = 8

    private static func ask(_ harness: Harness, _ messages: [[String: String]]) async throws -> String {
        var output = ""
        for try await delta in harness.session.streamChat(
            messages: messages, maxNew: turns, options: greedy
        ) { output += delta }
        return output
    }

    private static let q1 = [["role": "user", "content": "The capital of France is"]]
    private static func q2(after reply: String, _ first: [[String: String]] = q1) -> [[String: String]] {
        first + [
            ["role": "assistant", "content": reply],
            ["role": "user", "content": "and the capital of Spain is"],
        ]
    }

    /// Warm turn 2 (state from turn 1 reused) is byte-identical to turn 2 on
    /// a cold session, and the reuse actually happened: only the new turn's
    /// tail was fed.
    @Test func warmSecondTurnEqualsColdSecondTurn() async throws {
        let warm = try Self.makeHarness(think: false)
        let first = try await Self.ask(warm, Self.q1)
        #expect(first.count == Self.turns)
        let fedByTurnOne = warm.model.calls.reduce(0) { $0 + $1.count }
        let transcript = Self.q2(after: first)
        let reused = try await Self.ask(warm, transcript)
        let decision = warm.session.lastMetrics.prefixReuse
        #expect(decision.reused, "the second turn did not reuse the first turn's state: \(decision.reason)")
        #expect(decision.reusedTokens == fedByTurnOne)
        let prompt = Self.render(transcript, think: false)
        #expect(warm.model.calls.last(where: { $0.count > 1 }) == Array(prompt[fedByTurnOne...]),
                "only the tail past the match is fed")

        let cold = try Self.makeHarness(think: false)
        let fresh = try await Self.ask(cold, transcript)
        #expect(!cold.session.lastMetrics.prefixReuse.reused)
        #expect(Array(reused.utf8) == Array(fresh.utf8),
                "reusing the prefix changed the output: warm \(reused) vs cold \(fresh)")
    }

    /// A changed earlier message shares no usable prefix: the turn starts
    /// over (reason logged), and the output equals cold.
    @Test func changedEarlierMessageIsNotReused() async throws {
        let warm = try Self.makeHarness(think: false)
        let first = try await Self.ask(warm, Self.q1)
        let edited = [["role": "user", "content": "The capital of Italy is"]]
        let transcript = Self.q2(after: first, edited)
        let diverged = try await Self.ask(warm, transcript)
        let decision = warm.session.lastMetrics.prefixReuse
        #expect(!decision.reused)
        #expect(decision.reason.contains("diverges from the cached state at token"), Comment(rawValue: decision.reason))
        #expect(warm.model.calls.last(where: { $0.count > 1 }) == Self.render(transcript, think: false),
                "a diverging prompt is prefilled whole")

        let cold = try Self.makeHarness(think: false)
        let fresh = try await Self.ask(cold, transcript)
        #expect(Array(diverged.utf8) == Array(fresh.utf8),
                "a diverging prompt was contaminated by the previous turn")
    }

    /// With reuse off every turn is a cold prefill, and the output is the
    /// same bytes as with reuse on — the A/B that shows reuse changes nothing
    /// but the time.
    @Test func offSwitchGivesIdenticalOutputWithZeroReuse() async throws {
        let off = try Self.makeHarness(think: false)
        off.session.prefixReuseEnabled = false
        let first = try await Self.ask(off, Self.q1)
        let transcript = Self.q2(after: first)
        let withoutReuse = try await Self.ask(off, transcript)
        let decision = off.session.lastMetrics.prefixReuse
        #expect(decision.reusedTokens == 0)
        #expect(decision.reason.contains("disabled"), Comment(rawValue: decision.reason))
        #expect(off.model.calls.last(where: { $0.count > 1 }) == Self.render(transcript, think: false))

        let on = try Self.makeHarness(think: false)
        let firstOn = try await Self.ask(on, Self.q1)
        #expect(firstOn == first)
        let withReuse = try await Self.ask(on, transcript)
        #expect(on.session.lastMetrics.prefixReuse.reused)
        #expect(Array(withoutReuse.utf8) == Array(withReuse.utf8))

        let cold = try Self.makeHarness(think: false)
        let fresh = try await Self.ask(cold, transcript)
        #expect(Array(withoutReuse.utf8) == Array(fresh.utf8))
    }

    /// Thinking-family templates: the state holds the empty think block the
    /// session fed to keep reasoning off, the re-render of that assistant
    /// turn has none, so the ids diverge there and the prefix is invalidated.
    /// This is the deliberate choice — a reused state would answer from a
    /// context the cold prefill never builds.
    @Test func thinkBlockReRenderIsNotReused() async throws {
        let warm = try Self.makeHarness(think: true)
        let first = try await Self.ask(warm, Self.q1)
        let promptOne = Self.render(Self.q1, think: true) + [Self.thinkClose]
        #expect(warm.model.calls.first == promptOne, "the empty think block is fed after <think>")
        let transcript = Self.q2(after: first)
        let rerendered = try await Self.ask(warm, transcript)
        let decision = warm.session.lastMetrics.prefixReuse
        #expect(!decision.reused)
        // The ids agree up to `<|im_start|>assistant\n`; the state's next id is
        // the think block, the re-render's is the reply's first byte.
        #expect(decision.reason.contains("diverges from the cached state at token \(promptOne.count - 2)"),
                Comment(rawValue: decision.reason))
        let promptTwo = Self.render(transcript, think: true) + [Self.thinkClose]
        #expect(warm.model.calls.last(where: { $0.count > 1 }) == promptTwo)

        let cold = try Self.makeHarness(think: true)
        let fresh = try await Self.ask(cold, transcript)
        #expect(Array(rerendered.utf8) == Array(fresh.utf8))
    }

    /// A prompt equal to (or shorter than) what the state holds would need a
    /// rewind; it starts over instead, and produces the same bytes as before.
    @Test func promptThatDoesNotExtendTheStateIsNotReused() async throws {
        let harness = try Self.makeHarness(think: false)
        let first = try await Self.ask(harness, Self.q1)
        let again = try await Self.ask(harness, Self.q1)
        let decision = harness.session.lastMetrics.prefixReuse
        #expect(!decision.reused)
        #expect(decision.reason.contains("does not extend the cached state"), Comment(rawValue: decision.reason))
        #expect(Array(again.utf8) == Array(first.utf8))
    }

    /// A step that reports cancellation after advancing the state leaves a
    /// record that cannot vouch for it; the state is dropped and the next
    /// turn, even one that would have extended it, is a cold prefill.
    @Test func cancelledStepLeavesNoReusableState() async throws {
        let harness = try Self.makeHarness(think: false)
        let first = try await Self.ask(harness, Self.q1)
        let transcript = Self.q2(after: first)
        let cancellation = GenerationCancellation()
        // Turn 2 reuses the prefix (one prefill call), then is cut off inside
        // its second decode step.
        harness.model.cancelOnCall = (harness.model.calls.count + 3, cancellation)
        var partial = ""
        for try await delta in harness.session.streamChat(
            messages: transcript, maxNew: Self.turns, options: Self.greedy,
            cancellation: cancellation
        ) { partial += delta }
        #expect(harness.session.lastMetrics.finishReason == .cancelled)
        #expect(harness.session.lastMetrics.prefixReuse.reused)
        harness.model.cancelOnCall = nil

        // The same transcript extended by the partial reply would match the
        // fed ids exactly were the state still trusted; it is not.
        let retry = Self.q2(after: partial, transcript)
        _ = try await Self.ask(harness, retry)
        let decision = harness.session.lastMetrics.prefixReuse
        #expect(!decision.reused)
        #expect(decision.reason == "no cached state", Comment(rawValue: decision.reason))
    }
}
