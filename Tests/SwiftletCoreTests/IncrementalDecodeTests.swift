import Foundation
import Testing
@testable import SwiftletCore

/// Incremental decode (KV cache + conv tail + delta state) must produce the
/// same logits as one whole-sequence pass, for both split points and pure
/// token-by-token decoding.
@Suite struct IncrementalDecodeTests {
    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    @Test func incrementalMatchesWholeSequence() throws {
        let model = try QwenCPUModel(modelDir: Self.fixturesDir.appendingPathComponent("tiny-model"))
        let tokens = [1, 5, 9, 42, 7, 99, 3, 17, 64, 2]
        let V = model.config.vocabSize

        let full = try model.forward(tokens: tokens, captureLayers: false)
        let fullLast = Array(full.logits[(tokens.count - 1) * V..<tokens.count * V])

        // Prefill 6, then decode 4 one at a time.
        let state = model.makeQwenContext()
        var logits = try model.step(Array(tokens[0..<6]), context: state)
        for t in tokens[6...] { logits = try model.step([t], context: state) }
        #expect(state.position == tokens.count)

        var maxDiff: Float = 0
        for i in 0..<V { maxDiff = max(maxDiff, abs(logits[i] - fullLast[i])) }
        #expect(maxDiff < 1e-4, "incremental vs whole-sequence logits maxAbsDiff \(maxDiff)")

        // Pure token-by-token from scratch too.
        let s2 = model.makeQwenContext()
        var logits2 = [Float]()
        for t in tokens { logits2 = try model.step([t], context: s2) }
        maxDiff = 0
        for i in 0..<V { maxDiff = max(maxDiff, abs(logits2[i] - fullLast[i])) }
        #expect(maxDiff < 1e-4, "token-by-token vs whole-sequence logits maxAbsDiff \(maxDiff)")
    }

    @Test func greedyContinuationIsDeterministic() throws {
        let model = try QwenCPUModel(modelDir: Self.fixturesDir.appendingPathComponent("tiny-model-q4"))
        model.retainAllLayers = true
        let V = model.config.vocabSize

        func generate() throws -> [Int] {
            let state = model.makeQwenContext()
            var logits = try model.step([1, 5, 9], context: state)
            var out: [Int] = []
            for _ in 0..<8 {
                var best = 0
                for v in 1..<V where logits[v] > logits[best] { best = v }
                out.append(best)
                logits = try model.step([best], context: state)
            }
            return out
        }
        let a = try generate()
        let b = try generate()
        #expect(a == b)
        #expect(a.count == 8)
    }
}
