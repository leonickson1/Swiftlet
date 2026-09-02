import Foundation
import Testing
@testable import SwiftletCore

/// S7a: the public model protocol hands callers an opaque, model-owned
/// context instead of Qwen's decode state. These tests pin the seam from
/// both sides — a non-Qwen model can implement the protocol and drive the
/// shared generator, and the Qwen models honour the context lifecycle
/// bitwise.
@Suite struct InferenceContextTests {
    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    // MARK: - Protocol leak checks

    /// A stand-in backend that names no Qwen type anywhere. Its context is a
    /// plain counter and its "logits" are a deterministic function of the
    /// tokens fed so far, so the generator's output is predictable.
    final class CounterContext: InferenceContext {
        private(set) var position = 0
        var fed: [Int] = []
        func reset() { position = 0; fed = [] }
        func advance(_ tokens: [Int]) { fed += tokens; position += tokens.count }
    }

    final class CounterModel: InferenceModel {
        let vocabSize = 16
        // A directory that holds no generation_config.json, so the
        // generator's EOS lookup finds nothing (no stop tokens).
        let modelDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("swiftlet-counter-model-\(UUID().uuidString)")
        func makeContext() -> any InferenceContext { CounterContext() }
        func step(_ tokens: [Int], context: any InferenceContext) throws -> [Float] {
            guard let ctx = context as? CounterContext else {
                throw InferenceContextError.foreignContext
            }
            ctx.advance(tokens)
            // Next token = (sum of everything fed so far) mod vocab, as a
            // one-hot logit vector.
            var logits = [Float](repeating: 0, count: vocabSize)
            logits[ctx.fed.reduce(0, +) % vocabSize] = 1
            return logits
        }
    }

    /// Compile-level proof that the protocol is implementable without any
    /// Qwen type, plus a runtime check that the engine-agnostic generator
    /// really only uses the protocol: it must drive the stand-in end to end.
    @Test func protocolIsImplementableWithoutQwenTypes() throws {
        let model = CounterModel()
        let generator = TextGenerator(model: model)
        var out: [Int] = []
        try generator.generate(promptIds: [3, 4], maxNew: 4) { out.append($0); return true }
        // fed=[3,4] -> 7; fed+=[7] -> 14; fed+=[14] -> 28%16=12; fed+=[12] -> 40%16=8
        #expect(out == [7, 14, 12, 8])
    }

    /// The protocol declarations themselves must not mention a Qwen type.
    /// Reads the source because Swift has no reflection over protocol
    /// requirements; the file is small and holds nothing else.
    @Test func protocolDeclarationsNameNoQwenType() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/SwiftletCore/InferenceModel.swift")
        let text = try String(contentsOf: source, encoding: .utf8)
        #expect(text.contains("protocol InferenceModel"), "protocol file moved?")
        #expect(text.contains("protocol InferenceContext"), "protocol file moved?")
        #expect(!text.contains("Qwen"), "InferenceModel.swift names a Qwen type")
        #expect(!text.contains("DecodeState"), "InferenceModel.swift names the old decode state")
    }

    /// A context belongs to the model instance that made it. Handing it to
    /// another instance is refused rather than silently mixing state.
    @Test func foreignContextIsRefused() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let a = try QwenCPUModel(modelDir: dir)
        let b = try QwenCPUModel(modelDir: dir)
        let ctx = a.makeContext()
        _ = try a.step([1], context: ctx)
        #expect(throws: InferenceContextError.self) {
            _ = try b.step([5], context: ctx)
        }
        let gpuA = try QwenMetalModel(modelDir: dir)
        let gpuB = try QwenMetalModel(modelDir: dir)
        let gpuCtx = gpuA.makeContext()
        _ = try gpuA.step([1], context: gpuCtx)
        #expect(throws: InferenceContextError.self) {
            _ = try gpuB.step([5], context: gpuCtx)
        }
        // And a CPU context is not a Metal context.
        #expect(throws: InferenceContextError.self) {
            _ = try gpuA.step([5], context: ctx)
        }
    }

    // MARK: - Lifecycle

    /// create -> step -> reset -> step must reproduce a fresh context's
    /// logits bitwise on both engines: reset really discards KV rows, conv
    /// history, recurrence, and position.
    @Test func resetReproducesFreshContextBitwise() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let prompt = [1, 5, 9, 42]
        let follow = [7, 99]

        func run(_ model: any InferenceModel) throws -> (fresh: [[Float]], reused: [[Float]]) {
            let fresh = model.makeContext()
            var freshLogits = [try model.step(prompt, context: fresh)]
            for t in follow { freshLogits.append(try model.step([t], context: fresh)) }

            let reused = model.makeContext()
            _ = try model.step([3, 17, 64], context: reused)
            _ = try model.step([2], context: reused)
            #expect(reused.position == 4)
            reused.reset()
            #expect(reused.position == 0)
            var reusedLogits = [try model.step(prompt, context: reused)]
            for t in follow { reusedLogits.append(try model.step([t], context: reused)) }
            #expect(reused.position == prompt.count + follow.count)
            return (freshLogits, reusedLogits)
        }

        let cpu = try QwenCPUModel(modelDir: dir)
        cpu.retainAllLayers = true
        let cpuRuns = try run(cpu)
        #expect(cpuRuns.fresh == cpuRuns.reused, "CPU: reset context diverges from fresh")

        let gpu = try QwenMetalModel(modelDir: dir)
        let gpuRuns = try run(gpu)
        #expect(gpuRuns.fresh == gpuRuns.reused, "Metal: reset context diverges from fresh")
    }

    /// One model, two live contexts, stepped alternately, must produce the
    /// same logits and KV rows as each sequence run alone on that model.
    /// On the Metal fast path the bound context's conv history and delta
    /// recurrence live in GPU buffers, so a switch has to capture them into
    /// the outgoing context and restore the incoming one's (plus its KV
    /// rows, which the other context overwrote at the same positions).
    @Test func interleavedContextsMatchIsolatedRuns() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let a = [1, 5, 9, 42, 7, 99]
        let b = [3, 17, 64, 2, 11, 5]

        func isolated(_ model: any InferenceModel, _ seq: [Int]) throws
            -> (logits: [[Float]], context: QwenInferenceContext) {
            let ctx = model.makeContext()
            // Multi-token prefill then single-token decode: both step shapes.
            var out = [try model.step(Array(seq[0..<3]), context: ctx)]
            for t in seq[3...] { out.append(try model.step([t], context: ctx)) }
            return (out, ctx as! QwenInferenceContext)
        }

        func interleaved(_ model: any InferenceModel) throws
            -> (a: [[Float]], b: [[Float]], ctxA: QwenInferenceContext, ctxB: QwenInferenceContext) {
            let ctxA = model.makeContext()
            let ctxB = model.makeContext()
            var outA = [try model.step(Array(a[0..<3]), context: ctxA)]
            var outB = [try model.step(Array(b[0..<3]), context: ctxB)]
            for i in 3..<a.count {
                outA.append(try model.step([a[i]], context: ctxA))
                outB.append(try model.step([b[i]], context: ctxB))
            }
            return (outA, outB, ctxA as! QwenInferenceContext, ctxB as! QwenInferenceContext)
        }

        func check(_ model: any InferenceModel, label: String) throws {
            let refA = try isolated(model, a)
            let refB = try isolated(model, b)
            let mixed = try interleaved(model)
            #expect(mixed.a == refA.logits, "\(label): sequence A diverges when interleaved")
            #expect(mixed.b == refB.logits, "\(label): sequence B diverges when interleaved")
            #expect(mixed.ctxA.position == a.count && mixed.ctxB.position == b.count)
            for (layer, ref) in refA.context.kv {
                let got = try #require(mixed.ctxA.kv[layer], "\(label): A lost KV layer \(layer)")
                #expect(got.k == ref.k && got.v == ref.v, "\(label): A KV layer \(layer) diverged")
            }
            for (layer, ref) in refB.context.kv {
                let got = try #require(mixed.ctxB.kv[layer], "\(label): B lost KV layer \(layer)")
                #expect(got.k == ref.k && got.v == ref.v, "\(label): B KV layer \(layer) diverged")
            }
        }

        let cpu = try QwenCPUModel(modelDir: dir)
        cpu.retainAllLayers = true
        try check(cpu, label: "CPU")
        try check(try QwenMetalModel(modelDir: dir), label: "Metal")
    }
}
