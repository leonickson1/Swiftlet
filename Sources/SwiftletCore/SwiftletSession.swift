import Foundation
import Tokenizers

/// High-level chat session for app integration: owns the model, tokenizer,
/// chat templating, and metrics, and streams text deltas. This is the API an
/// app embeds (Priv AI's engines expose the same AsyncStream shape).
public final class SwiftletSession: @unchecked Sendable {
    public struct Metrics: Sendable {
        public var promptTokens = 0
        public var generatedTokens = 0
        public var timeToFirstToken: TimeInterval = 0
        public var tokensPerSecond: Double = 0
    }

    /// Sampling settings. Defaults follow Qwen's recommendation for
    /// non-thinking chat on quantized checkpoints (temperature 0.7, top-p 0.8,
    /// top-k 20), plus a presence penalty so greedy-style verbatim repetition
    /// of earlier replies can't happen.
    public struct GenerationOptions: Sendable {
        public var temperature: Float = 0.7
        public var topK: Int = 20
        public var topP: Float = 0.8
        /// One-time logit penalty for any token already generated (Qwen's
        /// recommendation for quantized checkpoints).
        public var presencePenalty: Float = 1.5
        /// Additional penalty that grows with each repetition — this is what
        /// actually breaks degenerate loops that a one-time penalty can't.
        public var frequencyPenalty: Float = 0.5
        /// EOS is banned until this many tokens exist. Quantized models given
        /// a "be concise" system prompt put real probability on stopping after
        /// 1-2 tokens ("I can" <eos>); a minimum length makes that impossible.
        public var minNew: Int = 8
        public init() {}
        public static var greedy: GenerationOptions {
            var o = GenerationOptions()
            o.temperature = 0
            o.presencePenalty = 0
            o.frequencyPenalty = 0
            return o
        }
    }

    public let modelDir: URL
    public let config: QwenConfig
    private let model: any InferenceModel
    private let generator: TextGenerator
    private let tokenizer: Tokenizer
    private let metricsLock = NSLock()
    private var _lastMetrics = Metrics()
    /// True when running on the Metal engine (vs the CPU reference fallback).
    public let usesGPU: Bool

    public var lastMetrics: Metrics {
        metricsLock.lock()
        defer { metricsLock.unlock() }
        return _lastMetrics
    }

    /// Loads tokenizer + model. Prefers the Metal streaming engine; falls back
    /// to the CPU reference if Metal is unavailable. `cacheBudgetGB` bounds the
    /// expert cache (keep small on iOS; jetsam is unforgiving).
    public init(modelDir: URL, retainAllLayers: Bool = false, cacheBudgetGB: Double = 2) async throws {
        self.modelDir = modelDir
        print("[SwiftletSession] loading tokenizer...")
        tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        print("[SwiftletSession] tokenizer ok; building Metal model...")
        if let gpu = try? QwenMetalModel(modelDir: modelDir, cacheBudgetGB: cacheBudgetGB) {
            model = gpu
            usesGPU = true
            print("[SwiftletSession] Metal model ready")
        } else {
            print("[SwiftletSession] Metal init FAILED, falling back to CPU (heavy)")
            let cpu = try QwenCPUModel(modelDir: modelDir)
            cpu.retainAllLayers = retainAllLayers
            model = cpu
            usesGPU = false
        }
        config = model.config
        generator = TextGenerator(model: model)
        // In no-think mode Qwen3.6 likes to emit stray <think>/</think>
        // tokens (sometimes mid-answer, truncating it if treated as a stop).
        // Ban them from sampling entirely: with reasoning disabled they are
        // never legitimate output, and banning them forces real content
        // until EOS. <|im_start|> is chat markup, never legal output either.
        var banned = Set<Int>()
        for tag in ["<think>", "</think>", "<|im_start|>"] {
            let ids = tokenizer.encode(text: tag)
            if ids.count == 1 { banned.formUnion(ids) }
        }
        suppressedIds = banned
        // Thinking-family templates (Qwen3.6) end the generation prompt with
        // "<think>\n"; instruct-family ones (Qwen3-Next-Instruct) don't.
        // Continuation turns must match, or the model sees think tags it was
        // never trained on and echoes chat markup into the reply.
        let probe = (try? tokenizer.applyChatTemplate(messages: [["role": "user", "content": "hi"]])) ?? []
        let thinkOpen = tokenizer.encode(text: "<think>\n")
        usesThinkPrompt = !thinkOpen.isEmpty && probe.count >= thinkOpen.count
            && Array(probe.suffix(thinkOpen.count)) == thinkOpen
    }

    private let suppressedIds: Set<Int>
    private let usesThinkPrompt: Bool

    // Conversation cache: the decode state persists across turns, so a
    // follow-up message only needs its own turn prefilled — without this,
    // every follow-up reprocesses the entire conversation (which at ~1 s/token
    // on a phone reads as "stuck on ..."). Whether the incoming messages
    // extend the cached conversation is decided by comparing message content,
    // not template tokens: the re-rendered template drops the think block from
    // past assistant turns, so token-prefix comparison always diverges.
    private var convState = QwenCPUModel.DecodeState()
    private var lastMessages: [[String: String]] = []
    private var lastReplyText = ""
    private var statePrimed = false

    /// Drops the cached conversation (e.g. when the user starts a new chat).
    public func resetConversation() {
        convState = QwenCPUModel.DecodeState()
        lastMessages = []
        lastReplyText = ""
        statePrimed = false
    }

    // Memory-pressure shrink coordination: shrinkCache replaces the expert
    // cache object, so it must NEVER run while a decode step is using it
    // (the race garbles expert reads — on iOS this showed up as corrupted
    // replies whenever a memory warning landed mid-generation). The warning
    // handler only sets a flag; the decode thread applies it between tokens.
    private let genLock = NSLock()
    private var generationActive = false
    private var pendingShrinkGB: Double?

    /// Frees most of the expert cache in response to OS memory pressure; it
    /// refills lazily as generation continues. Safe to call from any thread.
    public func handleMemoryPressure() {
        genLock.lock()
        if generationActive {
            pendingShrinkGB = 0.4
            genLock.unlock()
            print("[SwiftletSession] memory pressure: shrink deferred to between tokens")
        } else {
            defer { genLock.unlock() }
            (model as? QwenMetalModel)?.shrinkCache(toGB: 0.4)
            print("[SwiftletSession] memory pressure: cache shrunk (idle)")
        }
    }

    /// Applies a deferred pressure shrink; called on the decode thread
    /// between model steps, when no step is in flight.
    private func applyPendingShrink() {
        genLock.lock()
        let gb = pendingShrinkGB
        pendingShrinkGB = nil
        genLock.unlock()
        if let gb {
            (model as? QwenMetalModel)?.shrinkCache(toGB: gb)
            print("[SwiftletSession] deferred cache shrink applied")
        }
    }

    /// Full prompt for a fresh conversation, with reasoning disabled: the
    /// Qwen3.6 template ends the generation prompt with "<think>\n", which
    /// makes the model reason in the open — and since the literal <think>
    /// never appears in the *output*, downstream filters can't catch it.
    /// Appending the rest of an empty think block reproduces the template's
    /// own enable_thinking=false rendering byte for byte.
    private func freshPromptIds(_ messages: [[String: String]]) throws -> [Int] {
        var ids = try tokenizer.applyChatTemplate(messages: messages)
        // Compare token ids, not decoded text — decode can normalize or skip
        // special tokens and silently defeat a string comparison.
        if usesThinkPrompt {
            let thinkOpen = tokenizer.encode(text: "<think>\n")
            if ids.count >= thinkOpen.count, Array(ids.suffix(thinkOpen.count)) == thinkOpen {
                ids += tokenizer.encode(text: "\n</think>\n\n")
            }
        }
        return ids
    }

    /// If `messages` extends the conversation the model state already holds
    /// (same prior messages + our own last reply + one new user message),
    /// returns the tokens for just the new turn; nil means start clean.
    private func continuationIds(_ messages: [[String: String]]) -> [Int]? {
        guard statePrimed,
              messages.count == lastMessages.count + 2,
              let newUser = messages.last, newUser["role"] == "user",
              messages[messages.count - 2]["role"] == "assistant"
        else { return nil }
        for (a, b) in zip(lastMessages, messages)
        where a["role"] != b["role"] || a["content"] != b["content"] { return nil }
        let echoed = (messages[messages.count - 2]["content"] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard echoed == lastReplyText.trimmingCharacters(in: .whitespacesAndNewlines)
        else { return nil }
        // The state ends right after our reply text (EOS is never fed to the
        // model). Close the assistant turn, open the new user turn, and match
        // the template's generation-prompt style: thinking-family templates
        // get the empty think block (reasoning off), instruct ones don't.
        let turn = "<|im_end|>\n<|im_start|>user\n" + (newUser["content"] ?? "")
            + "<|im_end|>\n<|im_start|>assistant\n"
            + (usesThinkPrompt ? "<think>\n\n</think>\n\n" : "")
        return tokenizer.encode(text: turn)
    }

    /// One sampled token. Presence penalty pushes down everything generated
    /// so far; top-k/top-p/temperature otherwise standard.
    private func sample(_ logitsIn: [Float], options: GenerationOptions, seen: [Int: Int], banEOS: Bool) -> Int {
        var logits = logitsIn
        for t in suppressedIds { logits[t] = -.infinity }
        if banEOS {
            for t in generator.eosTokens { logits[t] = -.infinity }
        }
        if options.presencePenalty != 0 || options.frequencyPenalty != 0 {
            for (t, n) in seen {
                logits[t] -= options.presencePenalty + options.frequencyPenalty * Float(n)
            }
        }
        let vocab = min(config.vocabSize, logits.count)
        if options.temperature <= 0 {
            var best = 0
            for v in 1..<vocab where logits[v] > logits[best] { best = v }
            return best
        }
        // Partial top-k selection: fixed candidate set, replace the minimum.
        let k = max(1, min(options.topK, vocab))
        var cand: [Int] = Array(0..<k)
        var minPos = 0
        for i in 1..<k where logits[cand[i]] < logits[cand[minPos]] { minPos = i }
        for v in k..<vocab where logits[v] > logits[cand[minPos]] {
            cand[minPos] = v
            minPos = 0
            for i in 1..<k where logits[cand[i]] < logits[cand[minPos]] { minPos = i }
        }
        var top = cand.map { ($0, logits[$0]) }.sorted { $0.1 > $1.1 }
        var probs = top.map { expf(($0.1 - top[0].1) / options.temperature) }
        var sum = probs.reduce(0, +)
        if options.topP < 1 {
            var cum: Float = 0
            var cut = probs.count
            for i in 0..<probs.count {
                cum += probs[i] / sum
                if cum >= options.topP { cut = i + 1; break }
            }
            top = Array(top.prefix(cut))
            probs = Array(probs.prefix(cut))
            sum = probs.reduce(0, +)
        }
        var r = Float.random(in: 0..<1) * sum
        for i in 0..<probs.count {
            r -= probs[i]
            if r <= 0 { return top[i].0 }
        }
        return top[0].0
    }

    /// Applies the model's chat template to `messages` (role/content pairs,
    /// e.g. [["role": "user", "content": "hi"]]) and streams generated text
    /// deltas. Reasoning is disabled; the reply is the answer directly.
    /// Cancelling the consuming task stops generation.
    public func streamChat(
        messages: [[String: String]],
        maxNew: Int = 1024,
        options: GenerationOptions = GenerationOptions()
    ) -> AsyncThrowingStream<String, Swift.Error> {
        AsyncThrowingStream { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                self.genLock.lock()
                self.generationActive = true
                self.genLock.unlock()
                defer {
                    self.genLock.lock()
                    self.generationActive = false
                    self.genLock.unlock()
                    self.applyPendingShrink()
                }
                do {
                    let suffix: [Int]
                    if let delta = self.continuationIds(messages) {
                        suffix = delta
                    } else {
                        self.resetConversation()
                        suffix = try self.freshPromptIds(messages)
                    }

                    let start = Date()
                    var firstTokenAt: Date?
                    var generated: [Int] = []
                    var generatedCounts: [Int: Int] = [:]
                    var printed = ""
                    var cleanEnd = false

                    var logits = try self.model.step(suffix, state: self.convState)
                    let prefillDone = Date()

                    var decodeSeconds = 0.0
                    var stopReason = "maxNew"
                    for _ in 0..<maxNew {
                        self.applyPendingShrink()
                        // EOS is legal only for a reply that reads finished:
                        // long enough AND ending at a sentence or line break
                        // (a stop after "are:" mid-list is never valid). The
                        // 300-token escape keeps a punctuation-averse reply
                        // from running to maxNew.
                        let tail = printed.hasSuffix("\n") || {
                            guard let last = printed.trimmingCharacters(in: .whitespacesAndNewlines).last
                            else { return false }
                            return ".!?。！？".contains(last)
                        }()
                        let eosAllowed = generated.count >= options.minNew
                            && (tail || generated.count >= 300)
                        let best = self.sample(logits, options: options, seen: generatedCounts,
                                               banEOS: !eosAllowed)
                        if self.generator.eosTokens.contains(best) {
                            cleanEnd = true
                            stopReason = "eos"
                            break
                        }
                        if firstTokenAt == nil { firstTokenAt = Date() }
                        generated.append(best)
                        generatedCounts[best, default: 0] += 1
                        let text = self.tokenizer.decode(tokens: generated)
                        var terminated = false
                        // Streaming-safe delta: an emoji split across tokens
                        // decodes with a trailing U+FFFD that must be held
                        // back — printing it desyncs the prefix comparison
                        // and silences the rest of the turn.
                        if let (delta, newPrinted) = StreamingText.delta(printed: printed, decoded: text) {
                            if case .terminated = continuation.yield(delta) { terminated = true }
                            printed = newPrinted
                        }
                        let t0 = Date()
                        logits = try self.model.step([best], state: self.convState)
                        decodeSeconds += -t0.timeIntervalSinceNow
                        if terminated { cleanEnd = true; stopReason = "consumer-cancelled"; break }
                    }
                    // Flush whatever the hold-back logic still owes (a
                    // character completed by the last token, or a genuinely
                    // unfinishable U+FFFD when EOS cut an emoji short).
                    if let rest = StreamingText.finalDelta(
                        printed: printed, decoded: self.tokenizer.decode(tokens: generated)
                    ) {
                        continuation.yield(rest)
                    }
                    print("[SwiftletSession] stop: \(stopReason) after \(generated.count) tokens")
                    if !cleanEnd {
                        // Hit maxNew: state still matches the generated text,
                        // so the conversation remains continuable.
                        cleanEnd = true
                    }

                    self.lastMessages = messages
                    self.lastReplyText = self.tokenizer.decode(tokens: generated)
                    self.statePrimed = cleanEnd && !generated.isEmpty

                    self.metricsLock.lock()
                    self._lastMetrics = Metrics(
                        promptTokens: suffix.count,
                        generatedTokens: generated.count,
                        timeToFirstToken: (firstTokenAt ?? prefillDone).timeIntervalSince(start),
                        tokensPerSecond: decodeSeconds > 0 ? Double(generated.count) / decodeSeconds : 0
                    )
                    self.metricsLock.unlock()
                    continuation.finish()
                } catch {
                    self.resetConversation()
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
