import Foundation
import Testing
import Tokenizers
@testable import SwiftletCore

/// Port of colibri's `PrefixReuseContract` (`c/tests/test_inkling_prefix_serve.py`
/// over `c/kv_prefix.h`): reusing a previous turn's state must change nothing
/// but the time. Two turns on one session, the second reusing the first's
/// state, must produce byte-identical output to the second turn alone on a
/// cold session; rejection paths and a thinking-family re-render restored
/// from the assistant-header snapshot must equal cold as well.
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
        case "<|im_start|>assistant\n": return [imStart] + bytes("assistant") + [10]
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
    private final class HistoryModel: PersistableInferenceModel, @unchecked Sendable {
        var config: QwenConfig
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

        func snapshot(of state: QwenCPUModel.DecodeState) throws -> Data {
            lock.lock()
            defer { lock.unlock() }
            return try JSONEncoder().encode(histories[ObjectIdentifier(state)] ?? [])
        }

        func restoreState(from data: Data) throws -> QwenCPUModel.DecodeState {
            let history = try JSONDecoder().decode([Int].self, from: data)
            let state = QwenCPUModel.DecodeState()
            state.position = history.count
            lock.lock()
            histories[ObjectIdentifier(state)] = history
            lock.unlock()
            return state
        }

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

    /// The live state includes an empty think block that the next rendering
    /// drops. A snapshot after the assistant header must still match, and
    /// restoring it must produce the same answer as a cold full transcript.
    @Test func reRenderedThinkBlockReusesAssistantHeaderSnapshot() async throws {
        let warm = try Self.makeHarness(think: true)
        let first = try await Self.ask(warm, Self.q1)
        let promptOne = Self.render(Self.q1, think: true) + [Self.thinkClose]
        let answerStart = promptOne.count - 2
        #expect(warm.model.calls.first == Array(promptOne.prefix(answerStart)))
        #expect(warm.model.calls.dropFirst().first == [Self.thinkOpen, Self.thinkClose])
        let transcript = Self.q2(after: first)
        let rerendered = try await Self.ask(warm, transcript)
        let decision = warm.session.lastMetrics.prefixReuse
        #expect(decision.reusedTokens == answerStart, Comment(rawValue: decision.reason))
        #expect(decision.reason.contains("snapshot"), Comment(rawValue: decision.reason))
        let promptTwo = Self.render(transcript, think: true) + [Self.thinkClose]
        let secondPrefill = warm.model.calls.dropFirst(Self.turns + 2).prefix(2).flatMap { $0 }
        #expect(secondPrefill == Array(promptTwo.dropFirst(answerStart)))
        #expect(warm.session.lastMetrics.promptTokens == promptTwo.count - answerStart)

        let cold = try Self.makeHarness(think: true)
        let fresh = try await Self.ask(cold, transcript)
        #expect(Array(rerendered.utf8) == Array(fresh.utf8))
    }

    /// Exercise the real Tokenizers renderer/encoder, with a small local BPE
    /// vocabulary and each supported family's ChatML generation suffix. A
    /// wrong header or a boundary after the think opening loses this reuse.
    @Test(arguments: ["qwen3_next", "qwen3_5_moe", "qwen3_6_moe"])
    func assistantHeaderSnapshotMatchesTokenizerRendering(modelType: String) async throws {
        let thinking = modelType != "qwen3_next"
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }

        var vocab = ["[UNK]": 0, "<|im_start|>": 1, "<|im_end|>": 2, "<think>": 3, "</think>": 4]
        for scalar in [10] + Array(32...126) {
            vocab[String(Unicode.Scalar(scalar)!)] = vocab.count
        }
        let added: [[String: Any]] = ["<|im_start|>", "<|im_end|>", "<think>", "</think>"].map {
            ["id": vocab[$0]!, "content": $0, "single_word": false,
             "lstrip": false, "rstrip": false, "normalized": false, "special": true]
        }
        let tokenizerData: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocab, "merges": [String](), "unk_token": "[UNK]"],
            "added_tokens": added,
        ]
        // A deliberately small template, authored here rather than copied
        // from a checkpoint. Past assistant turns omit the think block.
        let template = "{% for m in messages %}{{ '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{% endfor %}"
            + "{{ '<|im_start|>assistant\\n' }}" + (thinking ? "{{ '<think>\\n' }}" : "")
        let tokenizerConfig: [String: Any] = [
            "tokenizer_class": "PreTrainedTokenizer", "unk_token": "[UNK]",
            "chat_template": template, "clean_up_tokenization_spaces": false,
        ]
        try JSONSerialization.data(withJSONObject: tokenizerData).write(to: folder.appendingPathComponent("tokenizer.json"))
        try JSONSerialization.data(withJSONObject: tokenizerConfig).write(to: folder.appendingPathComponent("tokenizer_config.json"))
        let tokenizer = try await AutoTokenizer.from(modelFolder: folder)
        let model = try HistoryModel(modelDir: Self.fixturesDir.appendingPathComponent("tiny-model"))
        model.config.modelType = modelType
        let session = SwiftletSession(
            testingModel: model, modelDir: model.modelDir,
            encodeText: { tokenizer.encode(text: $0) },
            decodeTokens: { tokenizer.decode(tokens: $0) },
            renderMessages: { try tokenizer.applyChatTemplate(messages: $0) },
            usesThinkPrompt: thinking
        )
        let harness = Harness(model: model, session: session)
        let first = try await Self.ask(harness, Self.q1)
        let rendered = try tokenizer.applyChatTemplate(messages: Self.q1)
        let header = tokenizer.encode(text: "<|im_start|>assistant\n")
        let openingCount = thinking ? tokenizer.encode(text: "<think>\n").count : 0
        let boundary = rendered.count - openingCount
        #expect(Array(rendered.prefix(boundary).suffix(header.count)) == header)
        #expect(model.calls.first == Array(rendered.prefix(boundary)))

        _ = try await Self.ask(harness, Self.q2(after: first))
        let decision = session.lastMetrics.prefixReuse
        #expect(decision.reusedTokens == (thinking ? boundary : boundary + Self.turns),
                Comment(rawValue: decision.reason))
    }

    @Test func thirdThinkingTurnUsesTheLatestSnapshot() async throws {
        let warm = try Self.makeHarness(think: true)
        let first = try await Self.ask(warm, Self.q1)
        let secondTranscript = Self.q2(after: first)
        let second = try await Self.ask(warm, secondTranscript)
        let thirdTranscript = Self.q2(after: second, secondTranscript)
        let third = try await Self.ask(warm, thirdTranscript)
        #expect(warm.session.lastMetrics.prefixReuse.reusedTokens == Self.render(secondTranscript, think: true).count - 1)
        let cold = try Self.makeHarness(think: true)
        #expect(try await Self.ask(cold, thirdTranscript) == third)
    }

    @Test func prefixMissReportsBothFollowCountsAndStopText() async throws {
        let harness = try Self.makeHarness(think: true)
        let first = try await Self.ask(harness, Self.q1)
        let edited = [["role": "user", "content": "The capital of Italy is"]]
        _ = try await Self.ask(harness, Self.q2(after: first, edited))
        let decision = harness.session.lastMetrics.prefixReuse
        #expect(!decision.reused)
        // Six ChatML/role ids, then the fifteen bytes "The capital of ".
        #expect(decision.reason.contains("followed the live state for 21, the snapshot for 21"),
                Comment(rawValue: decision.reason))
        #expect(decision.reason.contains("stopped at \"Italy isassis\""), Comment(rawValue: decision.reason))
    }

    @Test func thinkingOffSwitchAlsoDisablesSnapshots() async throws {
        let harness = try Self.makeHarness(think: true)
        harness.session.prefixReuseEnabled = false
        let first = try await Self.ask(harness, Self.q1)
        #expect(harness.model.calls.first == Self.render(Self.q1, think: true) + [Self.thinkClose])
        let transcript = Self.q2(after: first)
        let output = try await Self.ask(harness, transcript)
        #expect(!harness.session.lastMetrics.prefixReuse.reused)
        #expect(harness.session.lastMetrics.prefixReuse.reason.contains("disabled"))
        let cold = try Self.makeHarness(think: true)
        #expect(try await Self.ask(cold, transcript) == output)
    }

    @Test func rejectedSnapshotContinuationKeepsBothRecords() async throws {
        let harness = try Self.makeHarness(think: true)
        let first = try await Self.ask(harness, Self.q1)
        let original = harness.session.cachedConversationIds()
        let calls = harness.model.calls.count
        let tooLong = Self.q1 + [
            ["role": "assistant", "content": first],
            ["role": "user", "content": String(repeating: "x", count: 512)],
        ]
        await #expect(throws: ContextWindowError.self) {
            _ = try await Self.ask(harness, tooLong)
        }
        #expect(harness.session.cachedConversationIds() == original)
        #expect(harness.model.calls.count == calls)

        let transcript = Self.q2(after: first)
        let retry = try await Self.ask(harness, transcript)
        #expect(harness.session.lastMetrics.prefixReuse.restoredSnapshot)
        let cold = try Self.makeHarness(think: true)
        #expect(try await Self.ask(cold, transcript) == retry)
    }

    @Test func resetConversationDropsThePromptSnapshot() async throws {
        let harness = try Self.makeHarness(think: true)
        let first = try await Self.ask(harness, Self.q1)
        harness.session.resetConversation()
        _ = try await Self.ask(harness, Self.q2(after: first))
        #expect(!harness.session.lastMetrics.prefixReuse.reused)
        #expect(!harness.session.lastMetrics.prefixReuse.restoredSnapshot)
    }

    /// The installed Qwen3.6 tokenizer's vocabulary, merge order and template
    /// (reduced to the pieces this prompt uses), through Tokenizers itself.
    @Test func qwen36HeaderMatchesTheCheckpointTokenizerRendering() async throws {
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: Self.fixturesDir.appendingPathComponent("prompt-snapshot-tokenizer")
        )
        let question = [["role": "user", "content": "hi"]]
        let rendered = try tokenizer.applyChatTemplate(messages: question)
        #expect(rendered == [248045, 846, 198, 5834, 248046, 198, 248045, 74455, 198, 248068, 198])
        #expect(tokenizer.encode(text: "<|im_start|>assistant\n") == [248045, 74455, 198])
        let model = try HistoryModel(modelDir: Self.fixturesDir.appendingPathComponent("tiny-model"))
        model.config.modelType = "qwen3_5_moe"
        let session = SwiftletSession(
            testingModel: model, modelDir: model.modelDir,
            encodeText: { tokenizer.encode(text: $0) },
            decodeTokens: { tokenizer.decode(tokens: $0) },
            renderMessages: { try tokenizer.applyChatTemplate(messages: $0) }, usesThinkPrompt: true
        )
        for try await _ in session.streamChat(messages: question, maxNew: 0) {}
        #expect(model.calls.first == [248045, 846, 198, 5834, 248046, 198, 248045, 74455, 198])
        let transcript = question + [
            ["role": "assistant", "content": "hi"], ["role": "user", "content": "hi"],
        ]
        for try await _ in session.streamChat(messages: transcript, maxNew: 0) {}
        #expect(session.lastMetrics.prefixReuse.reusedTokens == 9)
        #expect(session.lastMetrics.prefixReuse.restoredSnapshot)
        #expect(!session.lastMetrics.prefixReuse.reason.contains("\n"), "stop text must stay on one log line")
    }

    /// Use actual CPU/Metal recurrence and SWLSTATE bytes as well as the
    /// history probe: restoring the session's seam must equal a cold render.
    @Test(arguments: [false, true])
    func thinkingSnapshotOnRealEnginesMatchesCold(useMetal: Bool) async throws {
        func makeSession() throws -> SwiftletSession {
            let folder = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
            let model: any InferenceModel = useMetal
                ? try QwenMetalModel(modelDir: folder) : try QwenCPUModel(modelDir: folder)
            return SwiftletSession(
                testingModel: model, modelDir: folder,
                encodeText: { Self.encode($0) },
                // Decimal ids expose every generated token, including those
                // which the text fixture's decoder normally hides.
                decodeTokens: { $0.map(String.init).joined(separator: ",") },
                renderMessages: { Self.render($0, think: true) }, usesThinkPrompt: true
            )
        }
        func ask(_ session: SwiftletSession, _ messages: [[String: String]]) async throws -> String {
            var output = ""
            for try await delta in session.streamChat(messages: messages, maxNew: 4, options: Self.greedy) {
                output += delta
            }
            return output
        }
        let warm = try makeSession()
        let first = try await ask(warm, Self.q1)
        let transcript = Self.q2(after: first)
        let resumed = try await ask(warm, transcript)
        #expect(warm.lastMetrics.prefixReuse.reusedTokens == 43)
        #expect(warm.lastMetrics.prefixReuse.restoredSnapshot)
        let cold = try makeSession()
        #expect(try await ask(cold, transcript) == resumed)
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
        #expect(decision.reason.hasPrefix("no cached state"), Comment(rawValue: decision.reason))
    }

    /// The record must be the session's own storage, never a view of the
    /// request's. FreeToken #559 had radix keys holding views into the
    /// request's id buffer, so a 64-byte key pinned the request's whole
    /// input+output allocation. Here the prompt arrives in a buffer reserved
    /// for a large output budget and is kept alive by the caller; after the
    /// turn the record holds exactly the fed ids, in storage sized to them
    /// (not the request's reservation) and at a different address.
    @Test func recordDoesNotRetainTheRequestBuffer() async throws {
        final class RequestBuffer: @unchecked Sendable {
            var ids: [Int] = []
            var base: UnsafeRawPointer?
        }
        let reserved = 1 << 17
        let request = RequestBuffer()
        let model = try HistoryModel(modelDir: Self.fixturesDir.appendingPathComponent("tiny-model"))
        let session = SwiftletSession(
            testingModel: model,
            modelDir: model.modelDir,
            encodeText: { Self.encode($0) },
            decodeTokens: { Self.decode($0) },
            renderMessages: { messages in
                var ids = Self.render(messages, think: false)
                ids.reserveCapacity(reserved)
                // Hold the buffer for the test's duration so the address
                // comparison below is between two live allocations.
                request.ids = ids
                request.base = ids.withUnsafeBufferPointer { UnsafeRawPointer($0.baseAddress) }
                return ids
            },
            usesThinkPrompt: false
        )
        let harness = Harness(model: model, session: session)
        let reply = try await Self.ask(harness, Self.q1)
        #expect(reply.count == Self.turns)
        #expect(request.ids.capacity >= reserved)

        let record = session.cachedConversationIds()
        #expect(record == Self.render(Self.q1, think: false) + Self.bytes(reply))
        #expect(
            record.capacity < reserved,
            "record capacity \(record.capacity) carries the request's reservation of \(reserved)"
        )
        let recordBase = record.withUnsafeBufferPointer { UnsafeRawPointer($0.baseAddress) }
        #expect(recordBase != nil)
        #expect(recordBase != request.base, "record storage aliases the request buffer")
    }
}
