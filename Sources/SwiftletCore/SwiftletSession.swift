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
        public var finishReason: GenerationFinishReason?
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
        /// Blocks any token that would repeat an already-emitted n-gram of this
        /// length. This is the deterministic loop-breaker greedy decoding needs:
        /// pure argmax on a quantized checkpoint otherwise falls into verbatim
        /// repetition cycles (issue #13), and penalties alone are probabilistic.
        /// 0 disables it; 3 is the standard safe value (short legitimate repeats
        /// in code, like indentation or operators, stay allowed).
        public var noRepeatNGram: Int = 3
        /// EOS is banned until this many tokens exist. Quantized models given
        /// a "be concise" system prompt put real probability on stopping after
        /// 1-2 tokens ("I can" <eos>); a minimum length makes that impossible.
        public var minNew: Int = 8
        /// Text sequences that end generation without being emitted. Potential
        /// prefixes are held across token boundaries, so split stop strings do
        /// not leak into a streamed response.
        public var stopSequences: [String] = []
        public init() {}
        public static var greedy: GenerationOptions {
            var o = GenerationOptions()
            o.temperature = 0
            // Greedy means a deterministic pick, not "no loop protection". The
            // frequency penalty and the n-gram block are both deterministic
            // functions of the generated history, so the pick stays fully
            // reproducible while degenerate repetition loops are broken (issue
            // #13: temperature 0 code prompts collapsed into verbatim cycles).
            // Only the flat presence penalty is dropped, because it distorts the
            // legitimate repetition in code (indentation, keywords, operators).
            o.presencePenalty = 0
            o.frequencyPenalty = 0.5
            o.noRepeatNGram = 3
            return o
        }
    }

    public let modelDir: URL
    public let config: QwenConfig
    private let model: any InferenceModel
    private let generator: TextGenerator
    private let encodeText: (String) -> [Int]
    private let decodeTokens: ([Int]) -> String
    private let renderMessages: ([[String: String]]) throws -> [Int]
    private let generationCleanupHook: (() -> Void)?
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
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDir)
        encodeText = { tokenizer.encode(text: $0) }
        decodeTokens = { tokenizer.decode(tokens: $0) }
        renderMessages = { try tokenizer.applyChatTemplate(messages: $0) }
        generationCleanupHook = nil
        print("[SwiftletSession] tokenizer ok; building Metal model...")
        do {
            let gpu = try QwenMetalModel(modelDir: modelDir, cacheBudgetGB: cacheBudgetGB)
            model = gpu
            usesGPU = true
            print("[SwiftletSession] Metal model ready")
        } catch {
            // A .qpack container streams its experts through the Metal cache
            // and the CPU reader cannot serve it, so an impossible cache
            // budget (or any other Metal failure) must surface here instead
            // of producing a model that fails on its first step.
            let isContainer = FileManager.default.fileExists(
                atPath: modelDir.appendingPathComponent("packed_experts/layout.json").path
            )
            if isContainer { throw error }
            print("[SwiftletSession] Metal init FAILED (\(error)), falling back to CPU (heavy)")
            let cpu = try QwenCPUModel(modelDir: modelDir)
            cpu.retainAllLayers = retainAllLayers
            model = cpu
            usesGPU = false
        }
        config = model.config
        generator = TextGenerator(model: model)
        // Special tokens (think tags, chat markup, FIM markers, vision and
        // tool tags) are never legitimate chat output, but a model pushed off
        // a momentarily-banned EOS will land on one of them: greedy 80B
        // emitted a literal <|fim_middle|> when its natural stop was gated.
        // Ban every added special token from sampling except the real stop
        // tokens (the same mechanism as HF's SuppressTokensLogitsProcessor).
        var banned = Set<Int>()
        let tokCfgURL = modelDir.appendingPathComponent("tokenizer_config.json")
        if let cfg = try? JSONSerialization.jsonObject(with: Data(contentsOf: tokCfgURL)) as? [String: Any],
           let added = cfg["added_tokens_decoder"] as? [String: Any] {
            for key in added.keys { if let id = Int(key) { banned.insert(id) } }
        }
        if banned.isEmpty {
            // No readable token config: fall back to banning the known-bad ones.
            for tag in ["<think>", "</think>", "<|im_start|>"] {
                let ids = tokenizer.encode(text: tag)
                if ids.count == 1 { banned.formUnion(ids) }
            }
        }
        banned.subtract(generator.eosTokens)
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

    /// Dependency-injected construction for deterministic session-boundary
    /// tests. Production callers use the async model/tokenizer initializer.
    init(
        testingModel model: any InferenceModel,
        modelDir: URL,
        encodeText: @escaping (String) -> [Int],
        decodeTokens: @escaping ([Int]) -> String,
        renderMessages: @escaping ([[String: String]]) throws -> [Int],
        suppressedIds: Set<Int> = [],
        usesThinkPrompt: Bool = false,
        generationCleanupHook: (() -> Void)? = nil
    ) {
        self.modelDir = modelDir
        self.model = model
        self.config = model.config
        self.generator = TextGenerator(model: model)
        self.usesGPU = model is QwenMetalModel
        self.encodeText = encodeText
        self.decodeTokens = decodeTokens
        self.renderMessages = renderMessages
        self.suppressedIds = suppressedIds
        self.usesThinkPrompt = usesThinkPrompt
        self.generationCleanupHook = generationCleanupHook
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
    /// A generation in flight is cancelled first, then the reset is ordered
    /// behind it on the generation queue so it cannot replace DecodeState while
    /// the model is using it. The caller therefore waits for at most one
    /// model-safe boundary (a CPU step or Metal command buffer), not for the
    /// remainder of the interrupted reply.
    public func resetConversation() {
        genLock.lock()
        let active = activeControl
        genLock.unlock()
        active?.cancel()
        generationQueue.sync {
            resetConversationState()
        }
    }

    /// The queue-confined form used by streamChat itself.
    private func resetConversationState() {
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

    /// Cancellation owner of the request currently executing on
    /// `generationQueue`, so `resetConversation()` can interrupt it instead of
    /// waiting for it to run to completion. Guarded by `genLock`.
    private var activeControl: GenerationCancellation?

    private func setActiveControl(_ control: GenerationCancellation?) {
        genLock.lock()
        activeControl = control
        genLock.unlock()
    }
    private var generationActive = false
    private var pendingShrinkGB: Double?
    /// Conversation state and the underlying model are both mutable. This is
    /// the session's single-flight boundary for every public streamChat call.
    private let generationQueue = DispatchQueue(
        label: "swiftlet.session.generation", qos: .userInitiated
    )

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

    private func beginGeneration() {
        genLock.lock()
        generationActive = true
        genLock.unlock()
    }

    /// Completes all shared lifecycle cleanup before the stream continuation
    /// is finished. Otherwise a waiting consumer can enqueue the next request
    /// while the preceding request still advertises stale active/shrink state.
    private func endGeneration() {
        // Keep advertising an active generation until every deferred shrink
        // has drained. A warning racing one shrink therefore queues another
        // pass instead of invoking shrinkCache concurrently.
        while true {
            genLock.lock()
            guard let gb = pendingShrinkGB else {
                generationActive = false
                genLock.unlock()
                break
            }
            pendingShrinkGB = nil
            genLock.unlock()
            (model as? QwenMetalModel)?.shrinkCache(toGB: gb)
            print("[SwiftletSession] deferred cache shrink applied")
        }
        generationCleanupHook?()
    }

    /// Full prompt for a fresh conversation, with reasoning disabled: the
    /// Qwen3.6 template ends the generation prompt with "<think>\n", which
    /// makes the model reason in the open — and since the literal <think>
    /// never appears in the *output*, downstream filters can't catch it.
    /// Appending the rest of an empty think block reproduces the template's
    /// own enable_thinking=false rendering byte for byte.
    private func freshPromptIds(_ messages: [[String: String]]) throws -> [Int] {
        var ids = try renderMessages(messages)
        // Compare token ids, not decoded text — decode can normalize or skip
        // special tokens and silently defeat a string comparison.
        if usesThinkPrompt {
            let thinkOpen = encodeText("<think>\n")
            if ids.count >= thinkOpen.count, Array(ids.suffix(thinkOpen.count)) == thinkOpen {
                ids += encodeText("\n</think>\n\n")
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
        return encodeText(turn)
    }

    /// Drops the replacement characters an incomplete trailing multi-byte
    /// sequence decodes to. Interior U+FFFD (a model genuinely emitting the
    /// replacement character) is preserved; only the unstable tail is held.
    /// Delegates to `StreamingText.stablePrefix` so the hold-back has one
    /// implementation instead of a second copy layered on the streaming path.
    static func trimIncompleteUTF8(_ s: String) -> String {
        StreamingText.stablePrefix(s)
    }

    /// Tokens that would complete an n-gram (length `n`) already emitted in
    /// `generated`. Banning them guarantees no verbatim n-gram — and therefore
    /// no verbatim repetition loop — can recur, which is the deterministic
    /// loop-breaker greedy decode needs (issue #13). Returns an empty set when
    /// `n` disables the guard or there is not yet a full n-gram of history.
    static func noRepeatNGramBanned(generated: [Int], n: Int) -> Set<Int> {
        guard n >= 2, generated.count >= n else { return [] }
        let prefix = Array(generated.suffix(n - 1))
        let lastStart = generated.count - (n - 1)   // exclusive: skips `prefix` itself
        var banned = Set<Int>()
        var i = 0
        while i < lastStart {
            if Array(generated[i..<i + (n - 1)]) == prefix {
                banned.insert(generated[i + (n - 1)])
            }
            i += 1
        }
        return banned
    }

    /// One sampled token. Presence penalty pushes down everything generated
    /// so far; top-k/top-p/temperature otherwise standard. `generated` is the
    /// ordered history, used only for the no-repeat-n-gram loop guard.
    private func sample(_ logitsIn: [Float], options: GenerationOptions,
                        generated: [Int], seen: [Int: Int], banEOS: Bool) -> Int {
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
        // Hard loop-breaker: forbid completing an n-gram that already occurred,
        // so no verbatim sequence can repeat (issue #13). Skipped in the
        // degenerate case where it would ban the entire candidate set.
        if options.noRepeatNGram >= 2 {
            let banned = Self.noRepeatNGramBanned(generated: generated, n: options.noRepeatNGram)
            if !banned.isEmpty && banned.count < vocab {
                for t in banned where t < logits.count { logits[t] = -.infinity }
            }
        }
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
    /// Cancelling the consuming task, or the optional cancellation owner,
    /// stops generation at the next model-safe boundary.
    public func streamChat(
        messages: [[String: String]],
        maxNew: Int = 1024,
        options: GenerationOptions = GenerationOptions(),
        cancellation: GenerationCancellation? = nil
    ) -> AsyncThrowingStream<String, Swift.Error> {
        AsyncThrowingStream { continuation in
            let control = cancellation ?? GenerationCancellation()
            continuation.onTermination = { termination in
                if case .cancelled = termination { control.cancel() }
            }
            self.generationQueue.async {
                self.beginGeneration()
                self.setActiveControl(control)
                var terminalError: Swift.Error?
                defer {
                    self.setActiveControl(nil)
                    self.endGeneration()
                    if let terminalError {
                        continuation.finish(throwing: terminalError)
                    } else {
                        continuation.finish()
                    }
                }
                do {
                    // A server request can be cancelled while waiting behind
                    // another request on the serial generation queue. Observe
                    // that before consulting or mutating conversation state,
                    // and report a fresh result instead of stale metrics from
                    // the preceding request.
                    if control.isCancelled {
                        self.storeMetrics(Metrics(finishReason: .cancelled))
                        return
                    }

                    let suffix: [Int]
                    if let delta = self.continuationIds(messages) {
                        suffix = delta
                    } else {
                        self.resetConversationState()
                        suffix = try self.freshPromptIds(messages)
                    }
                    // Admission runs before any model call, so a rejection
                    // leaves the cached conversation exactly as the client
                    // last saw it. Keep it: a shorter retry can then continue
                    // instead of re-prefilling the whole conversation.
                    let admittedMaxNew: Int
                    do {
                        admittedMaxNew = try ContextWindow(
                            maximumTokens: self.model.contextCapacity
                        ).admittedMaxNew(
                            processedTokens: self.convState.position,
                            incomingTokens: suffix.count,
                            requestedMaxNew: maxNew
                        )
                    } catch {
                        terminalError = error
                        return
                    }

                    let start = Date()
                    var firstTokenAt: Date?
                    var generated: [Int] = []
                    var generatedCounts: [Int: Int] = [:]
                    var stopFilter = StopSequenceFilter(stopSequences: options.stopSequences)
                    var matchedTextStop = false
                    var prefillDone = start
                    var decodeSeconds = 0.0
                    var finishReason = GenerationFinishReason.length

                    do {
                        var logits = try self.model.step(
                            suffix, state: self.convState,
                            shouldCancel: { control.isCancelled }
                        )
                        prefillDone = Date()

                        for _ in 0..<admittedMaxNew {
                            try checkGenerationCancellation { control.isCancelled }
                            self.applyPendingShrink()
                            try checkGenerationCancellation { control.isCancelled }
                            // EOS is legal when the reply reads finished (ends
                            // at a sentence or line break) OR EOS is the model's
                            // top choice outright. `stopFilter.output` is only
                            // text already safe to expose; a possible stop
                            // prefix must not influence this punctuation gate.
                            let visible = stopFilter.output
                            let tail = visible.hasSuffix("\n") || {
                                guard let last = visible.trimmingCharacters(in: .whitespacesAndNewlines).last
                                else { return false }
                                return ".!?。！？".contains(last)
                            }()
                            let eosIsTop = { () -> Bool in
                                let vocab = min(self.config.vocabSize, logits.count)
                                var best = 0
                                for v in 1..<vocab where logits[v] > logits[best] { best = v }
                                return self.generator.eosTokens.contains(best)
                            }()
                            let eosAllowed = generated.count >= options.minNew
                                && (tail || eosIsTop || generated.count >= 300)
                            let best = self.sample(
                                logits, options: options, generated: generated,
                                seen: generatedCounts, banEOS: !eosAllowed
                            )
                            if self.generator.eosTokens.contains(best) {
                                finishReason = .stop
                                break
                            }

                            if firstTokenAt == nil { firstTokenAt = Date() }
                            generated.append(best)
                            generatedCounts[best, default: 0] += 1

                            // Decode cumulatively so a stop can cross arbitrary
                            // token boundaries. The filter also owns the UTF-8
                            // and potential-stop-prefix hold-back.
                            let update = stopFilter.consume(
                                decoded: self.decodeTokens(generated)
                            )
                            guard update.isConsistent else {
                                throw GenerationInterruption.invalidTextStream
                            }
                            try checkGenerationCancellation { control.isCancelled }
                            if let delta = update.delta,
                               case .terminated = continuation.yield(delta) {
                                control.cancel()
                            }
                            if control.isCancelled {
                                finishReason = .cancelled
                                break
                            }
                            if update.didStop {
                                matchedTextStop = true
                                finishReason = .stop
                                break
                            }

                            let t0 = Date()
                            logits = try self.model.step(
                                [best], state: self.convState,
                                shouldCancel: { control.isCancelled }
                            )
                            decodeSeconds += -t0.timeIntervalSinceNow
                        }
                    } catch GenerationInterruption.cancelled {
                        finishReason = .cancelled
                    }

                    if control.isCancelled { finishReason = .cancelled }
                    // On EOS/length, inspect the final raw scalar stream before
                    // releasing held text. This both flushes a non-matching
                    // prefix/U+FFFD and catches a stop that itself ends in the
                    // now-terminal U+FFFD. Cancellation and prior matches do
                    // not release anything further.
                    if finishReason != .cancelled, !matchedTextStop {
                        let finalUpdate = stopFilter.finishUpdate(
                            decoded: self.decodeTokens(generated)
                        )
                        guard finalUpdate.isConsistent else {
                            throw GenerationInterruption.invalidTextStream
                        }
                        if finalUpdate.didStop {
                            matchedTextStop = true
                            finishReason = .stop
                        }
                        if let rest = finalUpdate.delta,
                           case .terminated = continuation.yield(rest) {
                            finishReason = .cancelled
                        }
                    }
                    // Cancellation can race the final EOS/length flush. Make
                    // the reuse decision only after one last observation so a
                    // turn cancelled during delivery cannot prime mismatched
                    // conversation state. Cancellation after this check is
                    // linearized as occurring after generation completed.
                    if control.isCancelled { finishReason = .cancelled }
                    print("[SwiftletSession] stop: \(finishReason.rawValue) after \(generated.count) tokens")

                    // EOS and length end with state exactly matching the text
                    // exposed to the caller. An arbitrary text stop may span
                    // already-fed token pieces, and cancellation can interrupt
                    // a prompt/token partway through, so neither is reusable
                    // without a real DecodeState snapshot/rollback facility.
                    let reusable = finishReason != .cancelled
                        && !matchedTextStop && !generated.isEmpty
                    if reusable {
                        self.lastMessages = messages
                        self.lastReplyText = stopFilter.output
                        self.statePrimed = true
                    } else {
                        self.resetConversationState()
                    }

                    self.storeMetrics(Metrics(
                        promptTokens: suffix.count,
                        generatedTokens: generated.count,
                        timeToFirstToken: (firstTokenAt ?? prefillDone).timeIntervalSince(start),
                        tokensPerSecond: decodeSeconds > 0 ? Double(generated.count) / decodeSeconds : 0,
                        finishReason: finishReason
                    ))
                } catch {
                    self.resetConversationState()
                    terminalError = error
                }
            }
        }
    }

    private func storeMetrics(_ metrics: Metrics) {
        metricsLock.lock()
        _lastMetrics = metrics
        metricsLock.unlock()
    }
}
