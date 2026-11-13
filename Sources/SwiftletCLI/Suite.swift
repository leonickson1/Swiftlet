import Foundation
import SwiftletCore
import Tokenizers

/// One line of a `--prompts-file` suite (JSONL). `prompt` is rendered through
/// the model's chat template with reasoning disabled, the way the app sends a
/// fresh conversation; `ids` bypasses the tokenizer (fixtures without one).
struct SuitePrompt: Decodable {
    let id: String
    let category: String?
    let prompt: String?
    let system: String?
    let ids: [Int]?
    let maxNew: Int?

    enum CodingKeys: String, CodingKey {
        case id, category, prompt, system, ids
        case maxNew = "max_new"
    }
}

/// One line of `--answers`: what the model said to one suite prompt and what
/// that prompt cost, with the expert cache counters for this prompt alone.
struct SuiteAnswer: Encodable {
    let index: Int
    let id: String
    let category: String
    let promptTokens: Int
    let generated: Int
    let hitEOS: Bool
    let prefillS: Double
    let decodeS: Double
    let decodeTokS: Double
    let hits: Int?
    let misses: Int?
    let text: String?
    let tokens: [Int]
}

func loadSuite(_ path: String) throws -> [SuitePrompt] {
    guard let data = FileManager.default.contents(atPath: path),
          let text = String(data: data, encoding: .utf8) else {
        throw CocoaError(.fileReadNoSuchFile, userInfo: [NSFilePathErrorKey: path])
    }
    var prompts: [SuitePrompt] = []
    for (n, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty || trimmed.hasPrefix("#") || trimmed.hasPrefix("//") { continue }
        do {
            let p = try JSONDecoder().decode(SuitePrompt.self, from: Data(trimmed.utf8))
            guard p.prompt != nil || p.ids != nil else {
                throw CocoaError(.coderValueNotFound, userInfo: [NSDebugDescriptionErrorKey: "needs prompt or ids"])
            }
            prompts.append(p)
        } catch {
            throw CocoaError(.coderReadCorrupt, userInfo: [
                NSDebugDescriptionErrorKey: "\(path) line \(n + 1): \(error)"])
        }
    }
    guard !prompts.isEmpty else {
        throw CocoaError(.coderValueNotFound, userInfo: [NSDebugDescriptionErrorKey: "\(path) has no prompts"])
    }
    return prompts
}

/// Token ids for a fresh conversation, matching SwiftletSession's
/// `freshPromptIds`: chat template, then the rest of an empty think block
/// when the template opens one (Qwen3.6), so the model answers directly.
/// A tokenizer without a chat template (the tiny fixtures) gets plain text.
func suitePromptIds(_ p: SuitePrompt, tokenizer: Tokenizer?) throws -> [Int] {
    if let ids = p.ids { return ids }
    guard let tokenizer, let prompt = p.prompt else {
        throw CocoaError(.coderValueNotFound, userInfo: [
            NSDebugDescriptionErrorKey: "prompt \(p.id) has no ids and the model has no tokenizer"])
    }
    var messages: [[String: String]] = []
    if let system = p.system { messages.append(["role": "system", "content": system]) }
    messages.append(["role": "user", "content": prompt])
    guard var ids = try? tokenizer.applyChatTemplate(messages: messages) else {
        FileHandle.standardError.write(Data("[\(p.id)] no chat template; encoding the prompt as plain text\n".utf8))
        return tokenizer.encode(text: prompt)
    }
    let thinkOpen = tokenizer.encode(text: "<think>\n")
    if !thinkOpen.isEmpty, ids.count >= thinkOpen.count, Array(ids.suffix(thinkOpen.count)) == thinkOpen {
        ids += tokenizer.encode(text: "\n</think>\n\n")
    }
    return ids
}

/// `swiftlet generate <model> --gpu --prompts-file suite.jsonl`: every prompt
/// in the file, in order, through one loaded model and therefore one expert
/// cache. Each prompt is its own conversation (fresh DecodeState, so the KV
/// and recurrent state reset), but the cache is never cleared, which is the
/// point: it measures how much of what one answer pulled in the next one
/// reuses. Decoding is greedy so two runs of a suite see identical routing.
func runSuite(
    modelDir: String, promptsFile: String, defaultMaxNew: Int,
    answersPath: String?, tracePath: String?
) async throws {
    let url = URL(fileURLWithPath: modelDir)
    let suite = try loadSuite(promptsFile)
    FileHandle.standardError.write(Data("suite: \(suite.count) prompts from \(promptsFile)\nloading tokenizer + model...\n".utf8))
    let needsTokenizer = suite.contains { $0.ids == nil }
    let tokenizer: Tokenizer? = needsTokenizer ? try await AutoTokenizer.from(modelFolder: url) : nil
    let model = try loadInferenceModel(url)
    let metal = model as? QwenMetalModel
    let eos = eosTokenIDs(url)

    var tracer: StepTracer?
    if let tracePath {
        if let metal {
            tracer = try StepTracer(
                path: tracePath, model: metal, modelDir: modelDir,
                promptTokens: nil, maxNew: nil, prompts: suite.count)
        }
        if tracer == nil {
            FileHandle.standardError.write(Data(
                "--trace-steps needs --gpu and a .qpack container; not tracing\n".utf8))
        }
    }

    var answersHandle: FileHandle?
    if let answersPath {
        guard FileManager.default.createFile(atPath: answersPath, contents: nil),
              let h = FileHandle(forWritingAtPath: answersPath) else {
            throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: answersPath])
        }
        answersHandle = h
    }
    defer { try? answersHandle?.close() }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    encoder.keyEncodingStrategy = .convertToSnakeCase

    let suiteStart = Date()
    var totalGenerated = 0
    var totalDecodeSeconds = 0.0
    for (index, p) in suite.enumerated() {
        let ids = try suitePromptIds(p, tokenizer: tokenizer)
        let maxNew = try ContextWindow(maximumTokens: model.contextCapacity)
            .admittedMaxNew(processedTokens: 0, incomingTokens: ids.count,
                            requestedMaxNew: p.maxNew ?? defaultMaxNew)
        let category = p.category ?? "uncategorized"
        tracer?.beginPrompt(index: index, id: p.id, category: category,
                            promptTokens: ids.count, maxNew: maxNew)
        let hitsBefore = metal?.expertCache?.hits
        let missesBefore = metal?.expertCache?.misses

        // A fresh state per prompt: position 0 resets the GPU-side recurrent
        // state; the expert cache is untouched.
        let state = QwenCPUModel.DecodeState()
        let prefillStart = Date()
        var logits = try model.step(ids, state: state)
        let prefillSecs = -prefillStart.timeIntervalSinceNow
        if let metal {
            tracer?.record(kind: "prefill", step: 0, token: nil, tokens: ids.count, metrics: metal.lastStepMetrics)
        }

        var generated: [Int] = []
        var hitEOS = false
        let decodeStart = Date()
        for _ in 0..<maxNew {
            var best = 0
            for v in 1..<model.config.vocabSize where logits[v] > logits[best] { best = v }
            if eos.contains(best) { hitEOS = true; break }
            generated.append(best)
            // Same loop as runGenerate, so a suite's decode records line up
            // one for one with a single-prompt trace of the same answer.
            logits = try model.step([best], state: state)
            if let metal {
                tracer?.record(kind: "decode", step: generated.count, token: best, tokens: 1,
                               metrics: metal.lastStepMetrics)
            }
        }
        let decodeSecs = -decodeStart.timeIntervalSinceNow
        tracer?.endPrompt(generated: generated.count, prefillSeconds: prefillSecs, decodeSeconds: decodeSecs)
        totalGenerated += generated.count
        totalDecodeSeconds += decodeSecs

        let text = tokenizer.map { $0.decode(tokens: generated) }
        let hits = hitsBefore.flatMap { before in metal?.expertCache.map { $0.hits - before } }
        let misses = missesBefore.flatMap { before in metal?.expertCache.map { $0.misses - before } }
        let tokS = Double(generated.count) / max(decodeSecs, 0.001)
        let answer = SuiteAnswer(
            index: index, id: p.id, category: category, promptTokens: ids.count,
            generated: generated.count, hitEOS: hitEOS, prefillS: prefillSecs, decodeS: decodeSecs,
            decodeTokS: tokS, hits: hits, misses: misses, text: text, tokens: generated)
        if let answersHandle {
            var data = try encoder.encode(answer)
            data.append(0x0A)
            answersHandle.write(data)
        }

        print("=== [\(index + 1)/\(suite.count)] \(p.id) (\(category)) ===")
        print(text ?? generated.map(String.init).joined(separator: " "))
        fflush(stdout)
        var line = String(format: "[%d/%d] %@: prefill %d tok %.1fs, decode %d tok %.1fs (%.2f tok/s)%@",
                          index + 1, suite.count, p.id, ids.count, prefillSecs,
                          generated.count, decodeSecs, tokS, hitEOS ? "" : " [hit max-new]")
        if let hits, let misses, hits + misses > 0 {
            line += String(format: ", cache %.0f%% hit (%d/%d)",
                           100 * Double(hits) / Double(hits + misses), hits, hits + misses)
        }
        FileHandle.standardError.write(Data((line + "\n").utf8))
    }
    tracer?.finish(generated: totalGenerated, decodeSeconds: totalDecodeSeconds)

    var summary = String(format: "suite: %d prompts, %d tokens, decode %.1fs (%.2f tok/s), wall %.1fs",
                         suite.count, totalGenerated, totalDecodeSeconds,
                         Double(totalGenerated) / max(totalDecodeSeconds, 0.001), -suiteStart.timeIntervalSinceNow)
    if let cache = metal?.expertCache {
        let total = cache.hits + cache.misses
        summary += String(format: "\nexpert cache: %d/%d slots, %d hits / %d misses (%.0f%% hit rate)",
                          cache.allocatedSlots, cache.slotCount, cache.hits, cache.misses,
                          total > 0 ? 100 * Double(cache.hits) / Double(total) : 0)
    }
    let memory = processMemory()
    summary += String(format: "\nmemory: peak footprint %.2f GiB, peak RSS %.2f GiB",
                      Double(memory.peakFootprint) / 1_073_741_824, Double(memory.peakRSS) / 1_073_741_824)
    FileHandle.standardError.write(Data((summary + "\n").utf8))
}

/// Peak physical footprint (what Activity Monitor calls Memory: dirty and
/// compressed pages, Metal buffers included, clean file cache excluded) and
/// peak resident set, for the RAM-versus-model-size claim.
func processMemory() -> (peakFootprint: Int, peakRSS: Int) {
    var usage = rusage()
    getrusage(RUSAGE_SELF, &usage)
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    let footprint = kr == KERN_SUCCESS ? Int(info.ledger_phys_footprint_peak) : 0
    return (footprint, Int(usage.ru_maxrss))
}
