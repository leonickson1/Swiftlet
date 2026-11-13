import Foundation
import SwiftletCore

/// `swiftlet generate ... --trace-steps <file>`: one JSON object per line.
///
/// A `run` header, then a `prefill` record and one `decode` record per
/// generated token, then an `end` record. Each step record carries the
/// step's S3a/S3c timing beside its expert fetches folded by tier (pool hit,
/// page cache miss, device miss) and the routing itself: which experts each
/// layer asked for, with a one letter outcome per expert (H hit, P page
/// cache miss, D device miss, U unprobed miss). That is enough to see where
/// in an answer the cache earns its RAM, and to replay the routing against
/// another policy offline. scripts/cache_ab.sh drives it; docs/CACHE_AB.md
/// explains the experiment.
///
/// A suite run (`--prompts-file`) writes one `run` header for the whole
/// process, then per prompt a `prompt` record, its `prefill` and `decode`
/// records tagged with the prompt's index, and a `prompt_end` record; the
/// cache carries over between prompts. scripts/cache_suite.sh drives it;
/// docs/CACHE_SUITE.md explains it.
final class StepTracer {
    private let handle: FileHandle
    private let sink = ExpertFetchTrace()
    private let cache: ExpertCache
    /// Index of the suite prompt in progress; nil outside a suite.
    private var currentPrompt: Int?
    private let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.sortedKeys]
        e.keyEncodingStrategy = .convertToSnakeCase
        return e
    }()

    struct RunRecord: Encodable {
        let kind = "run"
        let model: String
        let cacheGb: Double
        let slots: Int
        let strideBytes: Int
        let layers: Int
        let expertsPerLayer: Int
        let topK: Int
        /// Single-prompt runs only; a suite states these per `prompt` record.
        let promptTokens: Int?
        let maxNew: Int?
        /// Number of prompts in a suite run; absent for a single prompt.
        let prompts: Int?
        /// Whether misses can be split into page cache and device reads.
        let pageCacheProbe: Bool
        let pageSize: Int
        let started: String
    }

    struct LayerRecord: Encodable {
        /// Layer index.
        let l: Int
        /// Routed experts in request order.
        let e: [Int]
        /// One letter per expert: H, P, D or U (see the type comment).
        let o: String
    }

    struct PromptRecord: Encodable {
        let kind = "prompt"
        /// Position in the suite, from 0.
        let prompt: Int
        let id: String
        let category: String
        let promptTokens: Int
        let maxNew: Int
    }

    struct PromptEndRecord: Encodable {
        let kind = "prompt_end"
        let prompt: Int
        let generated: Int
        let prefillS: Double
        let decodeS: Double
        let hits: Int
        let misses: Int
        let allocatedSlots: Int
    }

    struct StepRecord: Encodable {
        let kind: String
        /// Suite runs only: the index of the prompt this step belongs to.
        let prompt: Int?
        let step: Int
        /// Token fed to a decode step; absent for prefill.
        let token: Int?
        let tokens: Int
        let wallS: Double
        let waitS: Double
        let encodeS: Double
        let gapS: Double
        let gpuS: Double
        /// Wall time inside the expert fetch scope (hits and misses).
        let fetchS: Double
        let fetches: Int
        let hits: Int
        let misses: Int
        let missPageCache: Int
        let missDevice: Int
        let missUnprobed: Int
        let residentMean: Double?
        /// Sum and max of per-blob pread wall time over the step's misses.
        let preadS: Double
        let preadMaxS: Double
        let preadMsHist: [Int]
        let cumulativeHits: Int
        let cumulativeMisses: Int
        let allocatedSlots: Int
        let layers: [LayerRecord]
    }

    struct EndRecord: Encodable {
        let kind = "end"
        let generated: Int
        let decodeS: Double
        let hits: Int
        let misses: Int
        let allocatedSlots: Int
    }

    /// Nil when the model has no expert cache (a raw checkpoint directory):
    /// there is nothing to trace, and the caller says so.
    init?(path: String, model: QwenMetalModel, modelDir: String,
          promptTokens: Int?, maxNew: Int?, prompts: Int? = nil) throws {
        guard let cache = model.expertCache else { return nil }
        self.cache = cache
        guard FileManager.default.createFile(atPath: path, contents: nil),
              let handle = FileHandle(forWritingAtPath: path) else {
            throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: path])
        }
        self.handle = handle
        cache.trace = sink
        try write(RunRecord(
            model: modelDir,
            cacheGb: Double(cache.budgetBytes) / 1_073_741_824,
            slots: cache.slotCount,
            strideBytes: cache.stride,
            layers: cache.layout.layerCount,
            expertsPerLayer: cache.layout.expertCount,
            topK: model.config.numExpertsPerTok,
            promptTokens: promptTokens,
            maxNew: maxNew,
            prompts: prompts,
            pageCacheProbe: cache.pageCacheProbeAvailable,
            pageSize: Int(getpagesize()),
            started: ISO8601DateFormatter().string(from: Date())
        ))
    }

    func record(kind: String, step: Int, token: Int?, tokens: Int, metrics: QwenMetalModel.StepMetrics) {
        let records = sink.drain()
        let summary = ExpertFetchTrace.StepSummary(records: records)
        let layers = records.map { record in
            LayerRecord(
                l: record.layer, e: record.experts,
                o: String(record.outcomes.map { outcome -> Character in
                    switch outcome {
                    case .hit: return "H"
                    case .miss(_, let resident):
                        guard let resident else { return "U" }
                        return resident >= 1 ? "P" : "D"
                    }
                }))
        }
        try? write(StepRecord(
            kind: kind, prompt: currentPrompt, step: step, token: token, tokens: tokens,
            wallS: metrics.stepWallSeconds,
            waitS: metrics.blockingWaitSeconds,
            encodeS: metrics.encodeSeconds,
            gapS: metrics.cpuGapSeconds,
            gpuS: metrics.gpuExecutionSeconds,
            fetchS: metrics.cpuGap.expertFetchSeconds,
            fetches: summary.fetches,
            hits: summary.hits,
            misses: summary.misses,
            missPageCache: summary.missesFromPageCache,
            missDevice: summary.missesFromDevice,
            missUnprobed: summary.missesUnprobed,
            residentMean: summary.residentFractionMean,
            preadS: summary.preadSeconds,
            preadMaxS: summary.preadMaxSeconds,
            preadMsHist: summary.preadHistogram,
            cumulativeHits: cache.hits,
            cumulativeMisses: cache.misses,
            allocatedSlots: cache.allocatedSlots,
            layers: layers
        ))
    }

    /// Opens a suite prompt: every step recorded until `endPrompt` carries
    /// its index.
    func beginPrompt(index: Int, id: String, category: String, promptTokens: Int, maxNew: Int) {
        // Anything fetched between prompts belongs to no step; drop it.
        _ = sink.drain()
        currentPrompt = index
        try? write(PromptRecord(
            prompt: index, id: id, category: category, promptTokens: promptTokens, maxNew: maxNew))
    }

    func endPrompt(generated: Int, prefillSeconds: Double, decodeSeconds: Double) {
        guard let index = currentPrompt else { return }
        try? write(PromptEndRecord(
            prompt: index, generated: generated, prefillS: prefillSeconds, decodeS: decodeSeconds,
            hits: cache.hits, misses: cache.misses, allocatedSlots: cache.allocatedSlots))
        currentPrompt = nil
    }

    func finish(generated: Int, decodeSeconds: Double) {
        try? write(EndRecord(
            generated: generated, decodeS: decodeSeconds,
            hits: cache.hits, misses: cache.misses, allocatedSlots: cache.allocatedSlots))
        cache.trace = nil
        try? handle.close()
    }

    private func write<T: Encodable>(_ record: T) throws {
        var data = try encoder.encode(record)
        data.append(0x0A)
        handle.write(data)
    }
}
