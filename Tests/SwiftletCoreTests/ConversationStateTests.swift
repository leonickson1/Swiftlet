import Foundation
import Testing
@testable import SwiftletCore

/// S6: a context can be written to a versioned binary snapshot and restored
/// into a fresh model instance, continuing bitwise; anything that does not
/// belong to this model, version, geometry, or dtype is refused.
@Suite struct ConversationStateTests {
    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")
    static let q4 = fixturesDir.appendingPathComponent("tiny-model-q4")
    static let f32 = fixturesDir.appendingPathComponent("tiny-model")

    static let prompt = [1, 5, 9, 42, 7]
    static let savedAfter = 4      // greedy tokens decoded before the snapshot
    static let continued = 5       // greedy tokens decoded after it

    static func argmax(_ v: [Float]) -> Int {
        var b = 0
        for i in 1..<v.count where v[i] > v[b] { b = i }
        return b
    }

    struct Trace: Equatable {
        var tokens: [Int] = []
        var logits: [[Float]] = []
    }

    /// Greedy-decodes `count` tokens from `logits`, recording every logits
    /// vector and pick.
    static func decode(_ model: any InferenceModel, _ ctx: any InferenceContext,
                       from logits: [Float], count: Int) throws -> (Trace, [Float]) {
        var trace = Trace()
        var logits = logits
        for _ in 0..<count {
            let t = argmax(logits)
            trace.tokens.append(t)
            logits = try model.step([t], context: ctx)
            trace.logits.append(logits)
        }
        return (trace, logits)
    }

    /// The uninterrupted reference: prompt, then savedAfter + continued
    /// greedy tokens on one context.
    static func uninterrupted(_ model: any InferenceModel) throws -> Trace {
        let ctx = model.makeContext()
        let first = try model.step(prompt, context: ctx)
        let (a, mid) = try decode(model, ctx, from: first, count: savedAfter)
        let (b, _) = try decode(model, ctx, from: mid, count: continued)
        return Trace(tokens: a.tokens + b.tokens, logits: a.logits + b.logits)
    }

    /// Same run, snapshotting after savedAfter tokens and finishing on a
    /// fresh model instance restored from the bytes.
    static func interrupted(
        _ model: any PersistableInferenceModel, fresh: () throws -> any PersistableInferenceModel
    ) throws -> (Trace, Data) {
        let ctx = model.makeContext()
        let first = try model.step(prompt, context: ctx)
        let (a, mid) = try decode(model, ctx, from: first, count: savedAfter)
        let data = try model.snapshot(of: ctx)
        let reopened = try fresh()
        let restored = try reopened.restoreContext(from: data)
        #expect(restored.position == ctx.position)
        let (b, _) = try decode(reopened, restored, from: mid, count: continued)
        return (Trace(tokens: a.tokens + b.tokens, logits: a.logits + b.logits), data)
    }

    // MARK: - Round trips

    @Test func cpuRoundTripIsBitwise() throws {
        let cpu = try QwenCPUModel(modelDir: Self.q4)
        cpu.retainAllLayers = true
        let reference = try Self.uninterrupted(cpu)
        let (resumed, _) = try Self.interrupted(cpu) {
            let m = try QwenCPUModel(modelDir: Self.q4)
            m.retainAllLayers = true
            return m
        }
        #expect(resumed.tokens == reference.tokens, "CPU: greedy tokens diverge after restore")
        #expect(resumed.logits == reference.logits, "CPU: logits diverge after restore")
        #expect(reference.tokens.count == Self.savedAfter + Self.continued)
    }

    @Test func metalRoundTripIsBitwise() throws {
        let gpu = try QwenMetalModel(modelDir: Self.q4)
        let reference = try Self.uninterrupted(gpu)
        let (resumed, _) = try Self.interrupted(gpu) { try QwenMetalModel(modelDir: Self.q4) }
        #expect(resumed.tokens == reference.tokens, "Metal: greedy tokens diverge after restore")
        #expect(resumed.logits == reference.logits, "Metal: logits diverge after restore")
    }

    /// The layer-major chunked prefill and a snapshot taken mid-prompt (no
    /// decode yet) must restore the same as the decode-time snapshot.
    @Test func metalSnapshotAfterChunkedPrefillRestoresBitwise() throws {
        let long = Array(repeating: Self.prompt, count: 3).flatMap { $0 }   // 15 tokens
        let gpu = try QwenMetalModel(modelDir: Self.q4)
        gpu.prefillMode = .layerMajor(chunkTokens: 4)                      // crosses chunks
        let ctx = gpu.makeContext()
        let first = try gpu.step(long, context: ctx)
        let (reference, _) = try Self.decode(gpu, ctx, from: first, count: 4)

        let ctx2 = gpu.makeContext()
        let first2 = try gpu.step(long, context: ctx2)
        let data = try gpu.snapshot(of: ctx2)
        let reopened = try QwenMetalModel(modelDir: Self.q4)
        let restored = try reopened.restoreContext(from: data)
        let (resumed, _) = try Self.decode(reopened, restored, from: first2, count: 4)
        #expect(resumed == reference, "Metal: chunked-prefill snapshot diverges after restore")
    }

    /// Which bytes the Metal snapshot holds, proven against the GPU: the KV
    /// sections equal a direct readback of the GPU KV rows (the mirror is a
    /// memcpy of them), and the conv tail / delta state sections equal a
    /// direct readback of the per-layer GPU buffers.
    @Test func metalSnapshotMatchesDirectGPUReadback() throws {
        let gpu = try QwenMetalModel(modelDir: Self.q4)
        let ctx = gpu.makeQwenContext()
        let first = try gpu.step(Self.prompt, context: ctx)
        _ = try Self.decode(gpu, ctx, from: first, count: 3)
        let position = ctx.position
        #expect(gpu.boundContext === ctx, "context is not the bound one")

        let snapshot = try ConversationState.decode(try gpu.snapshot(of: ctx))
        #expect(snapshot.position == position)
        var kvLayers = 0, deltaLayers = 0
        for li in 0..<gpu.config.numHiddenLayers {
            if gpu.config.isLinearLayer(li) {
                deltaLayers += 1
                let hist = try #require(gpu.fastLayers[li].hist)
                let state = try #require(gpu.fastLayers[li].state)
                let histFloats = Array(UnsafeBufferPointer(
                    start: hist.contents().bindMemory(to: Float.self, capacity: hist.length / 4),
                    count: hist.length / 4))
                let stateFloats = Array(UnsafeBufferPointer(
                    start: state.contents().bindMemory(to: Float.self, capacity: state.length / 4),
                    count: state.length / 4))
                #expect(snapshot.section(layer: li, kind: .convTail)?.values == histFloats,
                        "layer \(li): conv tail section is not the GPU history")
                #expect(snapshot.section(layer: li, kind: .deltaState)?.values == stateFloats,
                        "layer \(li): delta state section is not the GPU recurrence")
            } else {
                kvLayers += 1
                let rows = try #require(gpu.attentionKVCacheRows(layer: li, count: position))
                #expect(snapshot.section(layer: li, kind: .k)?.values == rows.k,
                        "layer \(li): K section is not the GPU KV rows")
                #expect(snapshot.section(layer: li, kind: .v)?.values == rows.v,
                        "layer \(li): V section is not the GPU KV rows")
            }
        }
        #expect(kvLayers > 0 && deltaLayers > 0)
        #expect(snapshot.sections.count == 2 * kvLayers + 2 * deltaLayers)
        // Snapshotting did not disturb the live context: the next step still
        // matches a never-snapshotted twin.
        let twin = gpu.makeQwenContext()
        let twinFirst = try gpu.step(Self.prompt, context: twin)
        let (twinTrace, _) = try Self.decode(gpu, twin, from: twinFirst, count: 3)
        let next = try gpu.step([twinTrace.tokens.last!], context: ctx)
        let twinNext = try gpu.step([twinTrace.tokens.last!], context: twin)
        #expect(next == twinNext, "snapshot disturbed the bound context")
    }

    /// The format is engine-neutral: a Metal snapshot restores into the CPU
    /// reference (same checkpoint) and continues within the engines' usual
    /// parity band. Not bitwise: the state itself carries GPU-vs-CPU
    /// accumulation noise.
    @Test func metalSnapshotRestoresIntoCPUReference() throws {
        let gpu = try QwenMetalModel(modelDir: Self.q4)
        let ctx = gpu.makeContext()
        let first = try gpu.step(Self.prompt, context: ctx)
        let (_, gpuLogits) = try Self.decode(gpu, ctx, from: first, count: 2)
        let data = try gpu.snapshot(of: ctx)

        let cpu = try QwenCPUModel(modelDir: Self.q4)
        cpu.retainAllLayers = true
        let restored = try cpu.restoreContext(from: data)
        let next = Self.argmax(gpuLogits)
        let cpuNext = try cpu.step([next], context: restored)
        let gpuNext = try gpu.step([next], context: ctx)
        let diff = MetalModelTests.maxAbsDiff(cpuNext, gpuNext)
        #expect(diff < 2e-3, "CPU continuation of a Metal snapshot maxAbsDiff \(diff)")
        #expect(Self.argmax(cpuNext) == Self.argmax(gpuNext))
    }

    // MARK: - Format pins

    @Test func headerLayoutIsPinned() throws {
        let cpu = try QwenCPUModel(modelDir: Self.q4)
        let ctx = cpu.makeContext()
        _ = try cpu.step(Self.prompt, context: ctx)
        let data = try cpu.snapshot(of: ctx)
        func u32(_ off: Int) -> UInt32 {
            data.subdata(in: off..<off + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        }
        func u64(_ off: Int) -> UInt64 {
            data.subdata(in: off..<off + 8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        }
        #expect(Array(data.prefix(8)) == Array("SWLSTATE".utf8))
        #expect(u32(8) == 1, "format version")
        #expect(u32(12) == 128, "header length")
        #expect(u32(16) == 1, "dtype float32")
        #expect(u32(20) == 2, "identity scheme: safetensors checkpoint")
        let cfg = cpu.config
        let geometry: [Int] = [
            cfg.numHiddenLayers, cfg.fullAttentionInterval, cfg.numKeyValueHeads, cfg.headDim,
            cfg.linearNumValueHeads, cfg.linearNumKeyHeads, cfg.linearKeyHeadDim,
            cfg.linearValueHeadDim, cfg.linearConvKernelDim, cfg.hiddenSize, cfg.vocabSize,
        ]
        for (i, g) in geometry.enumerated() {
            #expect(u32(56 + 4 * i) == UInt32(g), "geometry field \(i)")
        }
        #expect(u64(100) == UInt64(Self.prompt.count), "position")
        let attention = (0..<cfg.numHiddenLayers).filter { !cfg.isLinearLayer($0) }.count
        let linear = cfg.numHiddenLayers - attention
        #expect(u32(108) == UInt32(2 * attention + 2 * linear), "section count")
        #expect(u64(112) == UInt64(data.count), "total byte length")
        #expect(data.count > 128 + 32)
        // Same context, same bytes: the snapshot is deterministic.
        #expect(try cpu.snapshot(of: ctx) == data)
        // An identity that does not depend on which engine wrote it.
        let gpu = try QwenMetalModel(modelDir: Self.q4)
        let gctx = gpu.makeContext()
        _ = try gpu.step(Self.prompt, context: gctx)
        let gdata = try gpu.snapshot(of: gctx)
        #expect(gdata.subdata(in: 0..<100) == data.subdata(in: 0..<100),
                "CPU and Metal snapshots of one checkpoint disagree on identity or geometry")
    }

    @Test func emptyContextRoundTrips() throws {
        let cpu = try QwenCPUModel(modelDir: Self.q4)
        let data = try cpu.snapshot(of: cpu.makeContext())
        let restored = try cpu.restoreContext(from: data)
        #expect(restored.position == 0)
        let fresh = try cpu.step(Self.prompt, context: cpu.makeContext())
        #expect(try cpu.step(Self.prompt, context: restored) == fresh)
    }

    // MARK: - Refusals

    static func snapshotBytes() throws -> (QwenCPUModel, Data) {
        let cpu = try QwenCPUModel(modelDir: Self.q4)
        let ctx = cpu.makeContext()
        _ = try cpu.step(Self.prompt, context: ctx)
        return (cpu, try cpu.snapshot(of: ctx))
    }

    /// Rewrites bytes in the header and re-signs the trailer, so the refusal
    /// under test is the semantic one, not the digest.
    static func forged(_ data: Data, at offset: Int, u32 value: UInt32) -> Data {
        var body = data.subdata(in: 0..<data.count - 32)
        var le = value.littleEndian
        withUnsafeBytes(of: &le) { body.replaceSubrange(offset..<offset + 4, with: $0) }
        return ConversationState.signed(body)
    }

    @Test func refusesBadMagic() throws {
        let (cpu, data) = try Self.snapshotBytes()
        var bad = data
        bad.replaceSubrange(0..<8, with: Array("NOTSTATE".utf8))
        #expect(throws: ConversationStateError.badMagic) {
            _ = try cpu.restoreContext(from: bad)
        }
        #expect(throws: ConversationStateError.badMagic) {
            _ = try cpu.restoreContext(from: Data([1, 2, 3]))
        }
    }

    @Test func refusesUnsupportedVersion() throws {
        let (cpu, data) = try Self.snapshotBytes()
        let bad = Self.forged(data, at: 8, u32: 2)
        #expect(throws: ConversationStateError.unsupportedVersion(found: 2, supported: 1)) {
            _ = try cpu.restoreContext(from: bad)
        }
    }

    @Test func refusesTruncatedAndCorruptBytes() throws {
        let (cpu, data) = try Self.snapshotBytes()
        #expect(throws: ConversationStateError.truncated) {
            _ = try cpu.restoreContext(from: data.prefix(data.count - 1))
        }
        #expect(throws: ConversationStateError.truncated) {
            _ = try cpu.restoreContext(from: data.prefix(200))
        }
        var flipped = data
        flipped[200] ^= 0x40
        #expect(throws: ConversationStateError.corrupt("digest mismatch")) {
            _ = try cpu.restoreContext(from: flipped)
        }
    }

    @Test func refusesAnotherModel() throws {
        let (_, data) = try Self.snapshotBytes()
        // Same geometry, different checkpoint bytes (f32 vs int4).
        let other = try QwenCPUModel(modelDir: Self.f32)
        #expect(throws: ConversationStateError.self) {
            _ = try other.restoreContext(from: data)
        }
        do {
            _ = try other.restoreContext(from: data)
            Issue.record("restore into another checkpoint succeeded")
        } catch let error as ConversationStateError {
            guard case .modelMismatch = error else {
                Issue.record("expected modelMismatch, got \(error)")
                return
            }
        }
    }

    @Test func refusesGeometryMismatch() throws {
        let (cpu, data) = try Self.snapshotBytes()
        // vocabSize shapes no section, so this reaches the geometry compare.
        let vocab = cpu.config.vocabSize
        let bad = Self.forged(data, at: 56 + 4 * 10, u32: UInt32(vocab + 1))
        #expect(throws: ConversationStateError.geometryMismatch(
            field: "vocabSize", expected: vocab, found: vocab + 1)
        ) {
            _ = try cpu.restoreContext(from: bad)
        }
        // A layer count the sections cannot satisfy is refused structurally,
        // before any model is consulted.
        let layers = Self.forged(data, at: 56, u32: UInt32(cpu.config.numHiddenLayers + 1))
        #expect(throws: ConversationStateError.corrupt("layer 8 missing convTail")) {
            _ = try cpu.restoreContext(from: layers)
        }
    }

    @Test func refusesDtypeMismatch() throws {
        let (cpu, data) = try Self.snapshotBytes()
        let bad = Self.forged(data, at: 16, u32: 2)
        #expect(throws: ConversationStateError.dtypeMismatch(found: 2)) {
            _ = try cpu.restoreContext(from: bad)
        }
    }

    @Test func refusesSectionThatDoesNotFitPosition() throws {
        let (cpu, data) = try Self.snapshotBytes()
        // Claim one more position than the KV sections hold.
        let bad = Self.forged(data, at: 100, u32: UInt32(Self.prompt.count + 1))
        #expect(throws: ConversationStateError.self) {
            _ = try cpu.restoreContext(from: bad)
        }
    }

    @Test func refusesSnapshotOfForeignContext() throws {
        let a = try QwenCPUModel(modelDir: Self.q4)
        let b = try QwenCPUModel(modelDir: Self.q4)
        let ctx = a.makeContext()
        #expect(throws: InferenceContextError.foreignContext) {
            _ = try b.snapshot(of: ctx)
        }
    }
}
