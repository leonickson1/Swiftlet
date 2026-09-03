import Foundation
import SwiftletCore
import Tokenizers

func gib(_ bytes: Int) -> String { String(format: "%.2f GiB", Double(bytes) / Double(1 << 30)) }
func mib(_ bytes: Int) -> String { String(format: "%.1f MiB", Double(bytes) / Double(1 << 20)) }

/// S3a whole-run totals plus S3b phase/timeline folds. Encode-side cost is
/// exact per phase; wait/GPU cost is only available per command-buffer label
/// because one buffer can span several phases.
private struct MetalStepAggregate {
    var steps = 0
    var commandBuffersCommitted = 0
    var blockingWaits = 0
    var blockingWaitSeconds = 0.0
    var computeDispatchesEncoded = 0
    var commandBufferErrors = 0
    var gpuExecutionSeconds = 0.0
    var gpuTimedCommandBuffers = 0
    var gpuUntimedCommandBuffers = 0
    var phaseDispatches: [QwenMetalModel.StepPhase: Int] = [:]
    var phaseEncodeSeconds: [QwenMetalModel.StepPhase: Double] = [:]
    /// Per-phase GPU seconds; non-nil only when the device measured them
    /// (dispatch-boundary counters — see QwenMetalModel.PhaseGpuSplitSupport).
    var phaseGpuSeconds: [QwenMetalModel.StepPhase: Double]?
    var bufferCounts: [String: Int] = [:]
    var bufferWaitSeconds: [String: Double] = [:]
    var bufferGpuSeconds: [String: Double] = [:]
    var bufferGpuTimed: [String: Int] = [:]
    /// S3c: wall/encode/gap totals plus the gap sub-attribution.
    var stepWallSeconds = 0.0
    var encodeSeconds = 0.0
    var cpuGapSeconds = 0.0
    var gapEmbedding = 0.0
    var gapEmbeddingLookups = 0
    var gapRouter = 0.0
    var gapExpertFetch = 0.0
    var gapExpertHits = 0
    var gapExpertMisses = 0
    var gapKVMirror = 0.0
    var gapCommandBufferSetup = 0.0
    var gapCommit = 0.0
    var gapLogitsReadback = 0.0
    var gapOther = 0.0

    mutating func add(_ metrics: QwenMetalModel.StepMetrics) {
        steps += 1
        stepWallSeconds += metrics.stepWallSeconds
        encodeSeconds += metrics.encodeSeconds
        cpuGapSeconds += metrics.cpuGapSeconds
        gapEmbedding += metrics.cpuGap.embeddingSeconds
        gapEmbeddingLookups += metrics.cpuGap.embeddingLookups
        gapRouter += metrics.cpuGap.routerSeconds
        gapExpertFetch += metrics.cpuGap.expertFetchSeconds
        gapExpertHits += metrics.cpuGap.expertFetchHits
        gapExpertMisses += metrics.cpuGap.expertFetchMisses
        gapKVMirror += metrics.cpuGap.kvMirrorSeconds
        gapCommandBufferSetup += metrics.cpuGap.commandBufferSetupSeconds
        gapCommit += metrics.cpuGap.commitSeconds
        gapLogitsReadback += metrics.cpuGap.logitsReadbackSeconds
        gapOther += metrics.cpuGapOtherSeconds
        commandBuffersCommitted += metrics.commandBuffersCommitted
        blockingWaits += metrics.blockingWaits
        blockingWaitSeconds += metrics.blockingWaitSeconds
        computeDispatchesEncoded += metrics.computeDispatchesEncoded
        commandBufferErrors += metrics.commandBufferErrors
        gpuExecutionSeconds += metrics.gpuExecutionSeconds
        gpuTimedCommandBuffers += metrics.gpuTimedCommandBuffers
        gpuUntimedCommandBuffers += metrics.gpuUntimedCommandBuffers
        for (phase, n) in metrics.phaseDispatchesEncoded {
            phaseDispatches[phase, default: 0] += n
        }
        for (phase, s) in metrics.phaseEncodeSeconds {
            phaseEncodeSeconds[phase, default: 0] += s
        }
        if let measured = metrics.phaseGpuSeconds {
            var merged = phaseGpuSeconds ?? [:]
            for (phase, s) in measured { merged[phase, default: 0] += s }
            phaseGpuSeconds = merged
        }
        for sample in metrics.commandBufferTimeline {
            let label = sample.phases.map(\.rawValue).joined(separator: "+")
            bufferCounts[label, default: 0] += 1
            bufferWaitSeconds[label, default: 0] += sample.waitSeconds
            if let gpu = sample.gpuSeconds {
                bufferGpuSeconds[label, default: 0] += gpu
                bufferGpuTimed[label, default: 0] += 1
            }
        }
    }
}

/// S3b stderr lines: exact encode-side phase split, then wait/GPU folded by
/// command-buffer label (the honest granularity for those two costs).
private func phaseSummaryLines(_ prefix: String, _ agg: MetalStepAggregate) -> [String] {
    var lines: [String] = []
    let phases = QwenMetalModel.StepPhase.allCases.filter {
        agg.phaseDispatches[$0, default: 0] > 0 || agg.phaseEncodeSeconds[$0, default: 0] > 0
    }
    if !phases.isEmpty {
        let cells = phases.map { phase -> String in
            var cell = "\(phase.rawValue) enc="
                + String(format: "%.3fs", agg.phaseEncodeSeconds[phase, default: 0])
                + " disp=\(agg.phaseDispatches[phase, default: 0])"
            if let gpu = agg.phaseGpuSeconds {
                cell += String(format: " gpu=%.3fs", gpu[phase, default: 0])
            }
            return cell
        }
        lines.append("\(prefix) S3b encode phases: " + cells.joined(separator: " | "))
    }
    if !agg.bufferCounts.isEmpty {
        let cells = agg.bufferCounts.keys.sorted().map { label -> String in
            let timed = agg.bufferGpuTimed[label, default: 0]
            let gpu = timed > 0
                ? String(format: "%.3fs/%d timed", agg.bufferGpuSeconds[label, default: 0], timed)
                : "n/a"
            return "\(label) cb=\(agg.bufferCounts[label, default: 0])"
                + String(format: " wait=%.3fs", agg.bufferWaitSeconds[label, default: 0])
                + " gpu=\(gpu)"
        }
        lines.append("\(prefix) S3b buffers: " + cells.joined(separator: " | "))
    }
    if agg.steps > 0 {
        lines.append(String(format:
            "%@ S3c cpu-gap: wall=%.3fs wait=%.3fs encode=%.3fs gap=%.3fs | "
            + "embedding=%.3fs/%d | router=%.3fs | fetch=%.3fs/%d hit/%d miss | "
            + "kvMirror=%.3fs | cbSetup=%.3fs | commit=%.3fs | logits=%.3fs | other=%.3fs",
            prefix, agg.stepWallSeconds, agg.blockingWaitSeconds, agg.encodeSeconds,
            agg.cpuGapSeconds, agg.gapEmbedding, agg.gapEmbeddingLookups, agg.gapRouter,
            agg.gapExpertFetch, agg.gapExpertHits, agg.gapExpertMisses, agg.gapKVMirror,
            agg.gapCommandBufferSetup, agg.gapCommit, agg.gapLogitsReadback, agg.gapOther))
    }
    return lines
}

private func gpuTimingSummary(
    seconds: Double, timed: Int, untimed: Int
) -> String {
    timed > 0
        ? String(format: "%.3fs/%d timed/%d n/a", seconds, timed, untimed)
        : "n/a/\(untimed)cb"
}

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
        let metal = try QwenMetalModel(modelDir: url, cacheBudgetGB: cacheGB)
        // S1b prefill schedule knob: --prefill-chunk N sets the layer-major
        // chunk size; 0 restores the legacy token-major schedule (A/B runs).
        // Default (flag absent) is the model's layer-major default.
        if let chunkFlag = flagValue(CommandLine.arguments, "--prefill-chunk"),
           let chunk = Int(chunkFlag) {
            metal.prefillMode = chunk <= 0 ? .tokenMajor : .layerMajor(chunkTokens: chunk)
        }
        // S3b follow-up: what the device's counters can actually sample, and
        // whether the per-phase GPU split is measured or honestly absent.
        let split: String
        switch metal.phaseGpuSplitSupport {
        case .dispatchBoundaryCounters:
            split = "per-phase GPU split: measured (dispatch-boundary counters)"
        case .unsupported(let reason):
            split = "per-phase GPU split: unsupported (\(reason))"
        }
        FileHandle.standardError.write(Data(
            "S3b counters: \(metal.counterSamplingSupport.summary)\n\(split)\n".utf8))
        model = metal
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

    // Same admission as TextGenerator/SwiftletSession: the model rejects an
    // oversized prompt itself, but the output request must be clamped here or
    // the loop below runs into the context limit mid-reply and throws.
    let admittedMaxNew = try ContextWindow(maximumTokens: model.contextCapacity)
        .admittedMaxNew(processedTokens: 0, incomingTokens: ids.count, requestedMaxNew: maxNew)

    let state = QwenCPUModel.DecodeState()
    let prefillStart = Date()
    var logits = try model.step(ids, state: state)
    let prefillSecs = -prefillStart.timeIntervalSinceNow
    if let metal = model as? QwenMetalModel {
        let m = metal.lastStepMetrics
        let gpu = gpuTimingSummary(
            seconds: m.gpuExecutionSeconds,
            timed: m.gpuTimedCommandBuffers,
            untimed: m.gpuUntimedCommandBuffers
        )
        let line = String(format:
            "prefill S3a: %d tok wall=%.3fs wait=%.3fs/%d cb=%d err=%d dispatch=%d lm=%d skip=%d",
            ids.count, m.stepWallSeconds, m.blockingWaitSeconds, m.blockingWaits,
            m.commandBuffersCommitted, m.commandBufferErrors, m.computeDispatchesEncoded,
            m.logitProjections, m.avoidedLogitProjections
        ) + " gpu=\(gpu)\n"
        FileHandle.standardError.write(Data(line.utf8))
        var prefillAgg = MetalStepAggregate()
        prefillAgg.add(m)
        for extra in phaseSummaryLines("prefill", prefillAgg) {
            FileHandle.standardError.write(Data((extra + "\n").utf8))
        }
    } else {
        FileHandle.standardError.write(Data(String(
            format: "prefill: %d tokens in %.1fs\n", ids.count, prefillSecs
        ).utf8))
    }

    var generated: [Int] = []
    var printed = ""
    var decodeMetal = MetalStepAggregate()
    // S3c: the CLI's own per-token work sits in the bench's "wall" too;
    // report it beside the model's gap so the two are never confused.
    var cliArgmaxSeconds = 0.0
    var cliDetokenizeSeconds = 0.0
    let decodeStart = Date()
    for _ in 0..<admittedMaxNew {
        let argmaxStart = ProcessInfo.processInfo.systemUptime
        var best = 0
        for v in 1..<model.config.vocabSize where logits[v] > logits[best] { best = v }
        cliArgmaxSeconds += ProcessInfo.processInfo.systemUptime - argmaxStart
        if eos.contains(best) { break }
        generated.append(best)
        let detokStart = ProcessInfo.processInfo.systemUptime
        if let tokenizer {
            // Reprint only the stable delta so multi-byte tokens render
            // correctly; StreamingText holds back incomplete characters.
            let text = tokenizer.decode(tokens: generated)
            if let (delta, newPrinted) = StreamingText.delta(printed: printed, decoded: text) {
                print(delta, terminator: "")
                fflush(stdout)
                printed = newPrinted
            }
        } else {
            print(best, terminator: " ")
            fflush(stdout)
        }
        cliDetokenizeSeconds += ProcessInfo.processInfo.systemUptime - detokStart
        logits = try model.step([best], state: state)
        if let metal = model as? QwenMetalModel {
            decodeMetal.add(metal.lastStepMetrics)
        }
    }
    let decodeSecs = -decodeStart.timeIntervalSinceNow
    if let tokenizer,
       let rest = StreamingText.finalDelta(printed: printed, decoded: tokenizer.decode(tokens: generated)) {
        print(rest, terminator: "")
    }
    print("")
    FileHandle.standardError.write(Data(String(
        format: "decode: %d tokens in %.1fs (%.2f tok/s)\n",
        generated.count, decodeSecs, Double(generated.count) / max(decodeSecs, 0.001)
    ).utf8))
    if model is QwenMetalModel {
        let gpu = gpuTimingSummary(
            seconds: decodeMetal.gpuExecutionSeconds,
            timed: decodeMetal.gpuTimedCommandBuffers,
            untimed: decodeMetal.gpuUntimedCommandBuffers
        )
        let line = String(format:
            "decode Metal S3a: steps=%d cb=%d err=%d wait=%.3fs/%d dispatch=%d",
            decodeMetal.steps, decodeMetal.commandBuffersCommitted,
            decodeMetal.commandBufferErrors, decodeMetal.blockingWaitSeconds,
            decodeMetal.blockingWaits, decodeMetal.computeDispatchesEncoded
        ) + " gpu=\(gpu)\n"
        FileHandle.standardError.write(Data(line.utf8))
        for extra in phaseSummaryLines("decode", decodeMetal) {
            FileHandle.standardError.write(Data((extra + "\n").utf8))
        }
        FileHandle.standardError.write(Data(String(format:
            "decode S3c cli: argmax=%.3fs detokenize+print=%.3fs\n",
            cliArgmaxSeconds, cliDetokenizeSeconds).utf8))
    }
    if let metal = model as? QwenMetalModel, let cache = metal.expertCache {
        let total = cache.hits + cache.misses
        FileHandle.standardError.write(Data(String(
            format: "expert cache: %d/%d slots (%.1f GB logical, %.1f GB allocated / %.1f GB budget), %d hits / %d misses (%.0f%% hit rate)\n",
            cache.allocatedSlots, cache.slotCount,
            Double(cache.logicalBytes) / 1_073_741_824,
            Double(cache.allocatedBytes) / 1_073_741_824,
            Double(cache.budgetBytes) / 1_073_741_824,
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
    print("  swiftlet dump-tensor <model-dir> <module-path> <out.safetensors>   dequantized f32 weights of one module")
    print("  swiftlet generate <model-dir> --prompt \"...\" [--max-new 32] [--chat] [--gpu] [--cache-gb 8] [--prefill-chunk 32] [--lazy]")
    print("  swiftlet chat <model-dir> [\"turn\" ...] [--max-new 256] [--cache-gb 8] [--greedy] [--system \"...\"]")
    print("")
    print("  --gpu       Metal runtime; on a .qpack container experts stream through a bounded cache")
    print("  --cache-gb  expert cache budget, default 8 (note: swiftlet-server defaults to 2)")
    print("  --lazy      CPU path only: ~3 GB peak instead of ~10-14 GB, slower per step")
}
