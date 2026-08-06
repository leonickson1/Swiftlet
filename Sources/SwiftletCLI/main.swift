import Foundation
import SwiftletCore
import Tokenizers

func gib(_ bytes: Int) -> String { String(format: "%.2f GiB", Double(bytes) / Double(1 << 30)) }
func mib(_ bytes: Int) -> String { String(format: "%.1f MiB", Double(bytes) / Double(1 << 20)) }

func runInfo(_ modelKey: String) {
    guard let cfg = ArchConfig.known[modelKey] else {
        print("unknown model \(modelKey); known: \(ArchConfig.known.keys.sorted().joined(separator: ", "))")
        exit(1)
    }
    print("\(cfg.name)  [\(cfg.family.rawValue)]")
    print("  layers: \(cfg.layerCount) (\(cfg.linearLayerCount) DeltaNet + \(cfg.fullAttentionLayerCount) GQA), hidden \(cfg.hiddenSize), vocab \(cfg.vocabSize)")
    print("  experts: \(cfg.expertCount) x \(cfg.layerCount) layers = \(cfg.routedExpertTotal) blobs, top-\(cfg.expertTopK) + 1 shared")
    print("  expert blob (int4 g64): \(mib(cfg.expertBlobBytesInt4G64)) -> layer file ~\(gib(cfg.expertBlobBytesInt4G64 * cfg.expertCount))")
    print("  expert pool on SSD: ~\(gib(cfg.expertBlobBytesInt4G64 * cfg.routedExpertTotal))")
    print("  cold IO per token: \(cfg.routedFetchesPerToken) fetches, ~\(mib(cfg.expertBlobBytesInt4G64 * cfg.routedFetchesPerToken))")
    print("  KV per token (fp16, GQA layers only): \(cfg.kvBytesPerToken) B; 8K ctx = \(mib(cfg.kvBytesPerToken * 8192))")
    print("  DeltaNet fixed state: \(mib(cfg.deltaNetStateBytes))")
    print("  repack source: \(cfg.repackSource)")
}

/// Teacher-forcing comparison of the CPU reference against an mlx-generated
/// fixture file (scripts/gen_fixtures.py or scripts/gen_real_fixture.py).
func runVerify(modelDir: String, fixturesPath: String) throws {
    let start = Date()
    let fixtures = try SafetensorsFile(url: URL(fileURLWithPath: fixturesPath))
    let model = try QwenCPUModel(modelDir: URL(fileURLWithPath: modelDir))
    let tokens = try fixtures.ints("input_ids")
    print("model: \(modelDir)")
    print("tokens (\(tokens.count)): \(tokens)")

    let caps = try model.forward(tokens: tokens)

    func report(_ name: String, _ ours: [Float]) throws {
        guard fixtures.tensors[name] != nil else { return }
        let ref = try fixtures.floats(name)
        var maxDiff: Float = 0
        for i in 0..<min(ours.count, ref.count) { maxDiff = max(maxDiff, abs(ours[i] - ref[i])) }
        print(String(format: "  %-12@ maxAbsDiff %.3e", name as NSString, maxDiff))
    }

    try report("embed", caps.embed)
    for i in 0..<model.config.numHiddenLayers {
        let name = String(format: "layer_%02d", i)
        guard fixtures.tensors[name] != nil else { continue }
        try report(name, caps.layerOutputs[i])
    }

    // Layer-LOCAL check: run each layer from the REFERENCE input so upstream
    // drift can't pile up — this is what localizes a divergence to one layer.
    print("  --- layer-local (input from fixture) ---")
    for i in 0..<model.config.numHiddenLayers {
        let inName = i == 0 ? "embed" : String(format: "layer_%02d", i - 1)
        let outName = String(format: "layer_%02d", i)
        guard fixtures.tensors[inName] != nil, fixtures.tensors[outName] != nil else { continue }
        let refIn = try fixtures.floats(inName)
        let localOut = try model.layerForward(refIn, S: tokens.count, layerIndex: i)
        let refOut = try fixtures.floats(outName)
        var maxDiff: Float = 0
        var maxPos = 0
        for j in 0..<min(localOut.count, refOut.count) {
            let d = abs(localOut[j] - refOut[j])
            if d > maxDiff { maxDiff = d; maxPos = j / model.config.hiddenSize }
        }
        let kind = model.config.isLinearLayer(i) ? "delta" : "attn "
        print(String(format: "  local layer_%02d [%@] maxAbsDiff %.3e (worst at pos %d)", i, kind as NSString, maxDiff, maxPos))
    }
    try report("final_norm", caps.finalNorm)
    if fixtures.tensors["logits"] != nil {
        try report("logits", caps.logits)
    }
    if fixtures.tensors["logits_last"] != nil {
        let V = model.config.vocabSize
        let last = Array(caps.logits[(tokens.count - 1) * V..<tokens.count * V])
        try report("logits_last", last)
    }
    print("  greedy: \(caps.greedy)")

    let manifestPath = fixturesPath.replacingOccurrences(of: ".safetensors", with: ".json")
    if let data = FileManager.default.contents(atPath: manifestPath),
       let manifest = try JSONSerialization.jsonObject(with: data) as? [String: Any],
       let expected = manifest["greedy_next_tokens"] as? [Int] {
        print("  reference: \(expected)")
        print(caps.greedy == expected ? "  GREEDY MATCH" : "  GREEDY MISMATCH")
    }
    print(String(format: "  (%.1fs)", -start.timeIntervalSinceNow))
}

/// End-to-end text generation on the CPU reference runtime.
func runGenerate(modelDir: String, prompt: String, maxNew: Int, chat: Bool, rawIds: [Int]?) async throws {
    let url = URL(fileURLWithPath: modelDir)
    FileHandle.standardError.write(Data("loading tokenizer + model...\n".utf8))
    let tokenizer: Tokenizer? = rawIds == nil ? try await AutoTokenizer.from(modelFolder: url) : nil
    let model: any InferenceModel
    if CommandLine.arguments.contains("--gpu") {
        // Metal runtime: weights stay quantized; experts stream via the
        // bounded cache (--cache-gb) when the model is a .qpack container.
        let cacheGB = Double(flagValue(CommandLine.arguments, "--cache-gb") ?? "8") ?? 8
        model = try QwenMetalModel(modelDir: url, cacheBudgetGB: cacheGB)
    } else {
        let cpu = try QwenCPUModel(modelDir: url)
        // --lazy: ~3 GB peak instead of ~10-14 GB, at the cost of re-dequantizing
        // dense weights per step. Kind to machines that are doing other work.
        cpu.retainAllLayers = !CommandLine.arguments.contains("--lazy")
        model = cpu
    }

    var ids: [Int]
    if let rawIds {
        ids = rawIds
    } else if chat {
        ids = try tokenizer!.applyChatTemplate(messages: [["role": "user", "content": prompt]])
    } else {
        ids = tokenizer!.encode(text: prompt)
    }

    // EOS ids from generation_config.json / config.json (int or array).
    var eos = Set<Int>()
    for name in ["generation_config.json", "config.json"] {
        guard let data = FileManager.default.contents(atPath: url.appendingPathComponent(name).path),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { continue }
        if let v = obj["eos_token_id"] as? Int { eos.insert(v) }
        if let vs = obj["eos_token_id"] as? [Int] { eos.formUnion(vs) }
    }

    let state = QwenCPUModel.DecodeState()
    let prefillStart = Date()
    var logits = try model.step(ids, state: state)
    let prefillSecs = -prefillStart.timeIntervalSinceNow
    FileHandle.standardError.write(Data(String(format: "prefill: %d tokens in %.1fs\n", ids.count, prefillSecs).utf8))

    var generated: [Int] = []
    var printed = ""
    let decodeStart = Date()
    for _ in 0..<maxNew {
        var best = 0
        for v in 1..<model.config.vocabSize where logits[v] > logits[best] { best = v }
        if eos.contains(best) { break }
        generated.append(best)
        if let tokenizer {
            // Reprint only the stable delta so multi-byte tokens render correctly.
            let text = tokenizer.decode(tokens: generated)
            if text.hasPrefix(printed) {
                print(String(text.dropFirst(printed.count)), terminator: "")
                fflush(stdout)
                printed = text
            }
        } else {
            print(best, terminator: " ")
            fflush(stdout)
        }
        logits = try model.step([best], state: state)
    }
    let decodeSecs = -decodeStart.timeIntervalSinceNow
    print("")
    FileHandle.standardError.write(Data(String(
        format: "decode: %d tokens in %.1fs (%.2f tok/s)\n",
        generated.count, decodeSecs, Double(generated.count) / max(decodeSecs, 0.001)
    ).utf8))
    if let metal = model as? QwenMetalModel, let cache = metal.expertCache {
        let total = cache.hits + cache.misses
        FileHandle.standardError.write(Data(String(
            format: "expert cache: %d slots (%.1f GB), %d hits / %d misses (%.0f%% hit rate)\n",
            cache.slotCount, Double(cache.slotCount * cache.stride) / 1_073_741_824,
            cache.hits, cache.misses, total > 0 ? 100 * Double(cache.hits) / Double(total) : 0
        ).utf8))
    }
}

/// Multi-turn chat through SwiftletSession — the exact code path the app
/// uses (template + no-think prompt, conversation cache, sampling).
func runChat(modelDir: String, turns: [String], maxNew: Int, cacheGB: Double, greedy: Bool, system: String?) async throws {
    let session = try await SwiftletSession(
        modelDir: URL(fileURLWithPath: modelDir), cacheBudgetGB: cacheGB)
    var messages: [[String: String]] = []
    if let system { messages.append(["role": "system", "content": system]) }
    let options = greedy ? SwiftletSession.GenerationOptions.greedy
                         : SwiftletSession.GenerationOptions()
    for turn in turns {
        messages.append(["role": "user", "content": turn])
        print("\n>>> \(turn)")
        var reply = ""
        for try await delta in session.streamChat(messages: messages, maxNew: maxNew, options: options) {
            print(delta, terminator: "")
            fflush(stdout)
            reply += delta
        }
        print("")
        let m = session.lastMetrics
        FileHandle.standardError.write(Data(String(
            format: "[turn] prefill %d tok, ttft %.1fs, %d generated @ %.2f tok/s\n",
            m.promptTokens, m.timeToFirstToken, m.generatedTokens, m.tokensPerSecond
        ).utf8))
        messages.append(["role": "assistant", "content": reply])
    }
}

func flagValue(_ args: [String], _ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

let args = CommandLine.arguments
switch args.count >= 2 ? args[1] : "help" {
case "info" where args.count >= 3:
    runInfo(args[2])
case "verify" where args.count >= 4:
    do { try runVerify(modelDir: args[2], fixturesPath: args[3]) } catch {
        print("verify failed: \(error)")
        exit(1)
    }
case "dump-tensor" where args.count >= 5:
    // Debug: write the dequantized f32 weights of a module to a safetensors
    // file (swiftlet dump-tensor <model-dir> <module-path> <out.safetensors>).
    do {
        let ckpt = try Checkpoint(dir: URL(fileURLWithPath: args[2]))
        let path = args[3]
        let w = try ckpt.moduleWeight(path)
        let shape = try ckpt.shape(path + (ckpt.isQuantized(path) ? ".scales" : ".weight"))
        var logicalShape = shape
        if ckpt.isQuantized(path) {
            // scales shape = (rows..., groups); recover logical cols from count.
            let rows = shape.dropLast().reduce(1, *)
            logicalShape = Array(shape.dropLast()) + [w.count / rows]
        }
        let data = w.withUnsafeBufferPointer { Data(buffer: $0) }
        try SafetensorsFile.write(
            to: URL(fileURLWithPath: args[4]),
            tensors: [(name: path, dtype: "F32", shape: logicalShape, bytes: data)]
        )
        print("wrote \(path) shape \(logicalShape) (\(w.count) floats)")
    } catch {
        print("dump failed: \(error)")
        exit(1)
    }
case "generate" where args.count >= 3:
    do {
        try await runGenerate(
            modelDir: args[2],
            prompt: flagValue(args, "--prompt") ?? "The fieldfare is a bird that",
            maxNew: Int(flagValue(args, "--max-new") ?? "32") ?? 32,
            chat: args.contains("--chat"),
            rawIds: flagValue(args, "--ids").map { $0.split(separator: ",").compactMap { Int($0) } }
        )
    } catch {
        print("generate failed: \(error)")
        exit(1)
    }
case "chat" where args.count >= 3:
    do {
        // Positional args after the model dir (minus flags) are user turns.
        var turns: [String] = []
        var skip = false
        for a in args.dropFirst(3) {
            if skip { skip = false; continue }
            if a == "--max-new" || a == "--cache-gb" || a == "--system" { skip = true; continue }
            if a.hasPrefix("--") { continue }
            turns.append(a)
        }
        try await runChat(
            modelDir: args[2],
            turns: turns.isEmpty ? ["What is the capital of Spain?"] : turns,
            maxNew: Int(flagValue(args, "--max-new") ?? "256") ?? 256,
            cacheGB: Double(flagValue(args, "--cache-gb") ?? "8") ?? 8,
            greedy: args.contains("--greedy"),
            system: flagValue(args, "--system")
        )
    } catch {
        print("chat failed: \(error)")
        exit(1)
    }
default:
    print("usage:")
    print("  swiftlet info <model>            model budget summary (\(ArchConfig.known.keys.sorted().joined(separator: " | ")))")
    print("  swiftlet verify <model-dir> <fixtures.safetensors>   compare CPU forward vs mlx fixture")
    print("  swiftlet generate <model-dir> --prompt \"...\" [--max-new 32] [--chat]")
}
