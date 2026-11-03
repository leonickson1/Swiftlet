import Foundation
import Testing
@testable import SwiftletCore

/// A `DecodeState` can be written to a versioned binary snapshot and
/// restored into a fresh model instance, continuing bitwise; anything that
/// does not belong to this model, version, geometry, dtype, or capacity is
/// refused, and so is any live state a reader could not restore.
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
    static func decode(_ model: any InferenceModel, _ state: QwenCPUModel.DecodeState,
                       from logits: [Float], count: Int) throws -> (Trace, [Float]) {
        var trace = Trace()
        var logits = logits
        for _ in 0..<count {
            let t = argmax(logits)
            trace.tokens.append(t)
            logits = try model.step([t], state: state)
            trace.logits.append(logits)
        }
        return (trace, logits)
    }

    /// The uninterrupted reference: prompt, then savedAfter + continued
    /// greedy tokens on one state.
    static func uninterrupted(_ model: any InferenceModel) throws -> Trace {
        let state = QwenCPUModel.DecodeState()
        let first = try model.step(prompt, state: state)
        let (a, mid) = try decode(model, state, from: first, count: savedAfter)
        let (b, _) = try decode(model, state, from: mid, count: continued)
        return Trace(tokens: a.tokens + b.tokens, logits: a.logits + b.logits)
    }

    /// Same run, snapshotting after savedAfter tokens and finishing on a
    /// fresh model instance restored from the bytes.
    static func interrupted(
        _ model: any PersistableInferenceModel, fresh: () throws -> any PersistableInferenceModel
    ) throws -> (Trace, Data) {
        let state = QwenCPUModel.DecodeState()
        let first = try model.step(prompt, state: state)
        let (a, mid) = try decode(model, state, from: first, count: savedAfter)
        let data = try model.snapshot(of: state)
        let reopened = try fresh()
        let restored = try reopened.restoreState(from: data)
        #expect(restored.position == state.position)
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
        let state = QwenCPUModel.DecodeState()
        let first = try gpu.step(long, state: state)
        let (reference, _) = try Self.decode(gpu, state, from: first, count: 4)

        let state2 = QwenCPUModel.DecodeState()
        let first2 = try gpu.step(long, state: state2)
        let data = try gpu.snapshot(of: state2)
        let reopened = try QwenMetalModel(modelDir: Self.q4)
        let restored = try reopened.restoreState(from: data)
        let (resumed, _) = try Self.decode(reopened, restored, from: first2, count: 4)
        #expect(resumed == reference, "Metal: chunked-prefill snapshot diverges after restore")
    }

    /// Which bytes the Metal snapshot holds, proven against the GPU: the KV
    /// sections equal a direct readback of the GPU KV rows (the mirror is a
    /// memcpy of them), and the conv tail / delta state sections equal a
    /// direct readback of the per-layer GPU buffers.
    @Test func metalSnapshotMatchesDirectGPUReadback() throws {
        let gpu = try QwenMetalModel(modelDir: Self.q4)
        let state = QwenCPUModel.DecodeState()
        let first = try gpu.step(Self.prompt, state: state)
        _ = try Self.decode(gpu, state, from: first, count: 3)
        let position = state.position
        #expect(gpu.boundState === state, "state is not the bound one")

        let snapshot = try ConversationState.decode(try gpu.snapshot(of: state))
        #expect(snapshot.position == position)
        var kvLayers = 0, deltaLayers = 0
        for li in 0..<gpu.config.numHiddenLayers {
            if gpu.config.isLinearLayer(li) {
                deltaLayers += 1
                let hist = try #require(gpu.fastLayers[li].hist)
                let gpuState = try #require(gpu.fastLayers[li].state)
                let histFloats = Array(UnsafeBufferPointer(
                    start: hist.contents().bindMemory(to: Float.self, capacity: hist.length / 4),
                    count: hist.length / 4))
                let stateFloats = Array(UnsafeBufferPointer(
                    start: gpuState.contents().bindMemory(to: Float.self, capacity: gpuState.length / 4),
                    count: gpuState.length / 4))
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
        // Snapshotting did not disturb the live state: the next step still
        // matches a never-snapshotted twin.
        let twin = QwenCPUModel.DecodeState()
        let twinFirst = try gpu.step(Self.prompt, state: twin)
        let (twinTrace, _) = try Self.decode(gpu, twin, from: twinFirst, count: 3)
        let next = try gpu.step([twinTrace.tokens.last!], state: state)
        let twinNext = try gpu.step([twinTrace.tokens.last!], state: twin)
        #expect(next == twinNext, "snapshot disturbed the bound state")
    }

    /// The format is engine-neutral: a Metal snapshot restores into the CPU
    /// reference (same checkpoint) and continues within the engines' usual
    /// parity band. Not bitwise: the state itself carries GPU-vs-CPU
    /// accumulation noise.
    @Test func metalSnapshotRestoresIntoCPUReference() throws {
        let gpu = try QwenMetalModel(modelDir: Self.q4)
        let state = QwenCPUModel.DecodeState()
        let first = try gpu.step(Self.prompt, state: state)
        let (_, gpuLogits) = try Self.decode(gpu, state, from: first, count: 2)
        let data = try gpu.snapshot(of: state)

        let cpu = try QwenCPUModel(modelDir: Self.q4)
        cpu.retainAllLayers = true
        let restored = try cpu.restoreState(from: data)
        let next = Self.argmax(gpuLogits)
        let cpuNext = try cpu.step([next], state: restored)
        let gpuNext = try gpu.step([next], state: state)
        let diff = MetalModelTests.maxAbsDiff(cpuNext, gpuNext)
        #expect(diff < 2e-3, "CPU continuation of a Metal snapshot maxAbsDiff \(diff)")
        #expect(Self.argmax(cpuNext) == Self.argmax(gpuNext))
    }

    @Test func emptyStateRoundTrips() throws {
        let cpu = try QwenCPUModel(modelDir: Self.q4)
        let data = try cpu.snapshot(of: QwenCPUModel.DecodeState())
        let restored = try cpu.restoreState(from: data)
        #expect(restored.position == 0)
        let fresh = try cpu.step(Self.prompt, state: QwenCPUModel.DecodeState())
        #expect(try cpu.step(Self.prompt, state: restored) == fresh)
    }

    // MARK: - Format pins

    @Test func headerLayoutIsPinned() throws {
        let cpu = try QwenCPUModel(modelDir: Self.q4)
        let state = QwenCPUModel.DecodeState()
        _ = try cpu.step(Self.prompt, state: state)
        let data = try cpu.snapshot(of: state)
        func u32(_ off: Int) -> UInt32 {
            data.subdata(in: off..<off + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        }
        func u64(_ off: Int) -> UInt64 {
            data.subdata(in: off..<off + 8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        }
        #expect(Array(data.prefix(8)) == Array("SWLSTATE".utf8))
        #expect(u32(8) == 2, "format version")
        #expect(u32(12) == 144, "header length")
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
        #expect(u32(120) == UInt32(cfg.rotaryDims), "rotary dims")
        #expect(u32(124) == UInt32(cpu.contextCapacity), "writer context capacity")
        #expect(u64(128) == cfg.ropeTheta.bitPattern, "rope theta bit pattern")
        #expect(u64(136) == 0, "reserved")
        #expect(data.count > 144 + 32)
        // Same state, same bytes: the snapshot is deterministic.
        #expect(try cpu.snapshot(of: state) == data)
        // An identity that does not depend on which engine wrote it.
        let gpu = try QwenMetalModel(modelDir: Self.q4)
        let gstate = QwenCPUModel.DecodeState()
        _ = try gpu.step(Self.prompt, state: gstate)
        let gdata = try gpu.snapshot(of: gstate)
        #expect(gdata.subdata(in: 0..<136) == data.subdata(in: 0..<136),
                "CPU and Metal snapshots of one checkpoint disagree on identity, geometry, or rope fields")
    }

    @Test func infoReadsTheHeaderWithoutAModel() throws {
        let (cpu, data) = try Self.snapshotBytes()
        let info = try ConversationState.info(of: data)
        #expect(info.position == Self.prompt.count)
        #expect(info.writerContextCapacity == cpu.contextCapacity)
        #expect(info.identityScheme == 2)
        #expect(info.identityHex.count == 64)
        #expect(info.byteCount == data.count)
        #expect(throws: ConversationStateError.truncated) {
            _ = try ConversationState.info(of: data.prefix(data.count - 1))
        }
    }

    // MARK: - Refusals

    static func snapshotBytes() throws -> (QwenCPUModel, Data) {
        let cpu = try QwenCPUModel(modelDir: Self.q4)
        let state = QwenCPUModel.DecodeState()
        _ = try cpu.step(Self.prompt, state: state)
        return (cpu, try cpu.snapshot(of: state))
    }

    /// Rewrites bytes in the header and re-signs the trailer, so the refusal
    /// under test is the semantic one, not the digest.
    static func forged(_ data: Data, at offset: Int, u32 value: UInt32) -> Data {
        var body = data.subdata(in: 0..<data.count - 32)
        var le = value.littleEndian
        withUnsafeBytes(of: &le) { body.replaceSubrange(offset..<offset + 4, with: $0) }
        return ConversationState.signed(body)
    }

    static func forged(_ data: Data, at offset: Int, u64 value: UInt64) -> Data {
        var body = data.subdata(in: 0..<data.count - 32)
        var le = value.littleEndian
        withUnsafeBytes(of: &le) { body.replaceSubrange(offset..<offset + 8, with: $0) }
        return ConversationState.signed(body)
    }

    @Test func refusesBadMagic() throws {
        let (cpu, data) = try Self.snapshotBytes()
        var bad = data
        bad.replaceSubrange(0..<8, with: Array("NOTSTATE".utf8))
        #expect(throws: ConversationStateError.badMagic) {
            _ = try cpu.restoreState(from: bad)
        }
        #expect(throws: ConversationStateError.badMagic) {
            _ = try cpu.restoreState(from: Data([1, 2, 3]))
        }
    }

    /// The 128-byte v1 draft is not read: it reports its version, not a
    /// header-length corruption.
    @Test func refusesUnsupportedVersion() throws {
        let (cpu, data) = try Self.snapshotBytes()
        let bad = Self.forged(data, at: 8, u32: 1)
        #expect(throws: ConversationStateError.unsupportedVersion(found: 1, supported: 2)) {
            _ = try cpu.restoreState(from: bad)
        }
    }

    @Test func refusesTruncatedAndCorruptBytes() throws {
        let (cpu, data) = try Self.snapshotBytes()
        #expect(throws: ConversationStateError.truncated) {
            _ = try cpu.restoreState(from: data.prefix(data.count - 1))
        }
        #expect(throws: ConversationStateError.truncated) {
            _ = try cpu.restoreState(from: data.prefix(200))
        }
        var flipped = data
        flipped[200] ^= 0x40
        #expect(throws: ConversationStateError.corrupt("digest mismatch")) {
            _ = try cpu.restoreState(from: flipped)
        }
    }

    @Test func refusesAnotherModel() throws {
        let (_, data) = try Self.snapshotBytes()
        // Same geometry, different checkpoint bytes (f32 vs int4).
        let other = try QwenCPUModel(modelDir: Self.f32)
        do {
            _ = try other.restoreState(from: data)
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
            _ = try cpu.restoreState(from: bad)
        }
        // A layer count the sections cannot satisfy is refused structurally,
        // before any model is consulted.
        let layers = Self.forged(data, at: 56, u32: UInt32(cpu.config.numHiddenLayers + 1))
        #expect(throws: ConversationStateError.corrupt("layer 8 missing convTail")) {
            _ = try cpu.restoreState(from: layers)
        }
    }

    /// The K rows are cached post-rotation, so a snapshot is only meaningful
    /// under the same rotary split and RoPE base.
    @Test func refusesRopeMismatch() throws {
        let (cpu, data) = try Self.snapshotBytes()
        let rot = cpu.config.rotaryDims
        let badDims = Self.forged(data, at: 120, u32: UInt32(rot + 1))
        #expect(throws: ConversationStateError.geometryMismatch(
            field: "rotaryDims", expected: rot, found: rot + 1)
        ) {
            _ = try cpu.restoreState(from: badDims)
        }
        let theta = cpu.config.ropeTheta
        let badTheta = Self.forged(data, at: 128, u64: (theta / 10).bitPattern)
        #expect(throws: ConversationStateError.ropeThetaMismatch(expected: theta, found: theta / 10)) {
            _ = try cpu.restoreState(from: badTheta)
        }
        // The writer's capacity is informational: a different value restores.
        let otherWriter = Self.forged(data, at: 124, u32: UInt32(cpu.contextCapacity * 2))
        #expect(try cpu.restoreState(from: otherWriter).position == Self.prompt.count)
    }

    @Test func refusesDtypeMismatch() throws {
        let (cpu, data) = try Self.snapshotBytes()
        let bad = Self.forged(data, at: 16, u32: 2)
        #expect(throws: ConversationStateError.dtypeMismatch(found: 2)) {
            _ = try cpu.restoreState(from: bad)
        }
    }

    @Test func refusesSectionThatDoesNotFitPosition() throws {
        let (cpu, data) = try Self.snapshotBytes()
        // Claim one more position than the KV sections hold.
        let bad = Self.forged(data, at: 100, u32: UInt32(Self.prompt.count + 1))
        #expect(throws: ConversationStateError.self) {
            _ = try cpu.restoreState(from: bad)
        }
    }

    /// A structurally valid snapshot whose position exceeds the restoring
    /// engine's context capacity is refused after identity and geometry; a
    /// position equal to the capacity restores, and admission refuses the
    /// next token as it does today. (A real step cannot produce the first
    /// case: validateStep refuses it, so the bytes are built directly.)
    @Test func refusesPositionBeyondCapacity() throws {
        let (cpu, data) = try Self.snapshotBytes()
        let real = try ConversationState.decode(data)
        func zeroed(position: Int) throws -> Data {
            let g = real.geometry
            var sections: [ConversationState.Section] = []
            for li in 0..<g.numHiddenLayers {
                if g.isLinearLayer(li) {
                    sections.append(.init(layer: li, kind: .convTail,
                                          values: [Float](repeating: 0, count: try g.convTailFloats())))
                    sections.append(.init(layer: li, kind: .deltaState,
                                          values: [Float](repeating: 0, count: try g.deltaStateFloats())))
                } else {
                    let rows = [Float](repeating: 0, count: position * (try g.kvRowFloats()))
                    sections.append(.init(layer: li, kind: .k, values: rows))
                    sections.append(.init(layer: li, kind: .v, values: rows))
                }
            }
            var forged = real
            forged.position = position
            forged.sections = sections
            return ConversationState.encode(forged)
        }
        let capacity = cpu.contextCapacity
        #expect(throws: ConversationStateError.capacityExceeded(position: capacity + 1, capacity: capacity)) {
            _ = try cpu.restoreState(from: try zeroed(position: capacity + 1))
        }
        let full = try cpu.restoreState(from: try zeroed(position: capacity))
        #expect(full.position == capacity)
        #expect(throws: ContextWindowError.capacityExceeded(processed: capacity, incoming: 1, maximum: capacity)) {
            _ = try cpu.step([1], state: full)
        }
    }

    /// Header fields are attacker-chosen bytes (the trailer digest is free to
    /// compute), so they must be refused, never trapped on. Every geometry
    /// field at 0xFFFFFFFF is outside the v2 range; fields at the 2^24
    /// ceiling pass the range check but their derived section sizes overflow
    /// and are refused as corrupt; an out-of-range section count is refused
    /// before the table is read.
    @Test func refusesOverflowingGeometryWithoutTrapping() throws {
        let (cpu, data) = try Self.snapshotBytes()
        var saturated = data
        for i in 0..<11 { saturated = Self.forged(saturated, at: 56 + 4 * i, u32: UInt32.max) }
        #expect(throws: ConversationStateError.corrupt("geometry field out of range")) {
            _ = try cpu.restoreState(from: saturated)
        }
        var ceiling = data
        for i in 0..<11 { ceiling = Self.forged(ceiling, at: 56 + 4 * i, u32: UInt32(1 << 24)) }
        #expect(throws: ConversationStateError.corrupt("geometry overflow")) {
            _ = try cpu.restoreState(from: ceiling)
        }
        #expect(throws: ConversationStateError.corrupt("section count \(UInt32.max)")) {
            _ = try cpu.restoreState(from: Self.forged(data, at: 108, u32: UInt32.max))
        }
        // rotaryDims is only compared, never multiplied: a saturated value
        // reaches the model comparison and is refused there.
        #expect(throws: ConversationStateError.geometryMismatch(
            field: "rotaryDims", expected: cpu.config.rotaryDims, found: Int(UInt32.max))
        ) {
            _ = try cpu.restoreState(from: Self.forged(data, at: 120, u32: UInt32.max))
        }
    }

    /// The largest in-range position times the largest in-range KV row width
    /// does not fit a 64-bit section size: refused as corrupt, not trapped.
    /// The DeltaNet layers before the first attention layer still decode, so
    /// the refusal is the K section's size computation itself.
    @Test func refusesOverflowingPositionProductWithoutTrapping() throws {
        let (cpu, data) = try Self.snapshotBytes()
        var bad = Self.forged(data, at: 56 + 4 * 2, u32: UInt32(1 << 24))   // numKeyValueHeads
        bad = Self.forged(bad, at: 56 + 4 * 3, u32: UInt32(1 << 24))         // headDim
        bad = Self.forged(bad, at: 100, u64: UInt64(Int.max / 4))            // position
        #expect(throws: ConversationStateError.corrupt("geometry overflow")) {
            _ = try cpu.restoreState(from: bad)
        }
    }

    /// While a Metal instance holds a state bound, its conv tail and delta
    /// state live in that instance's GPU buffers and the CPU fields are a
    /// stale copy: only that instance may write it. Once it releases the
    /// state (a switch), every engine writes the same bytes.
    @Test func refusesStateBoundOnAnotherInstance() throws {
        let gpuA = try QwenMetalModel(modelDir: Self.q4)
        let state = QwenCPUModel.DecodeState()
        _ = try gpuA.step(Self.prompt, state: state)
        #expect(state.boundOn === gpuA)
        let cpu = try QwenCPUModel(modelDir: Self.q4)
        let gpuB = try QwenMetalModel(modelDir: Self.q4)
        let refusal = ConversationStateError.inconsistentState("state is bound on another engine instance")
        #expect(throws: refusal) { _ = try cpu.snapshot(of: state) }
        #expect(throws: refusal) { _ = try gpuB.snapshot(of: state) }
        let fromA = try gpuA.snapshot(of: state)

        _ = try gpuA.step([1], state: QwenCPUModel.DecodeState())
        #expect(state.boundOn == nil, "switch did not release the state")
        #expect(try cpu.snapshot(of: state) == fromA, "released state writes differently on the CPU engine")
        #expect(try gpuB.snapshot(of: state) == fromA, "released state writes differently on another Metal instance")
    }

    /// A CPU step cancelled after its first layer leaves the state
    /// half-advanced (that layer's KV rows or recurrence updated, position
    /// not); such a state is refused by the writer. A cancellation before any
    /// layer ran leaves the state untouched and writable.
    @Test func refusesStateLeftByACancelledCPUStep() throws {
        let cpu = try QwenCPUModel(modelDir: Self.q4)
        cpu.retainAllLayers = true
        let state = QwenCPUModel.DecodeState()
        _ = try cpu.step(Self.prompt, state: state)
        let before = try cpu.snapshot(of: state)

        // Checks run: once on entry, then before every layer. The third one
        // is before layer 1, so layer 0 has already run.
        var calls = 0
        #expect(throws: GenerationInterruption.self) {
            _ = try cpu.step([7], state: state, shouldCancel: { calls += 1; return calls == 3 })
        }
        #expect(state.tainted, "cancelling after the first layer did not taint the state")
        #expect(state.position == Self.prompt.count)
        #expect(throws: ConversationStateError.inconsistentState("step threw mid-token")) {
            _ = try cpu.snapshot(of: state)
        }

        let clean = QwenCPUModel.DecodeState()
        _ = try cpu.step(Self.prompt, state: clean)
        var n = 0
        #expect(throws: GenerationInterruption.self) {
            _ = try cpu.step([7], state: clean, shouldCancel: { n += 1; return n == 1 })
        }
        #expect(!clean.tainted)
        #expect(try cpu.snapshot(of: clean) == before, "a cancellation before any layer changed the state")
    }

    /// The same on the Metal fast path, cancelling between command buffers:
    /// the first layer's buffer has run, so its GPU recurrence advanced while
    /// later layers' did not. Neither this instance nor another may write or
    /// load the state.
    @Test func refusesStateLeftByACancelledMetalStep() throws {
        let gpu = try QwenMetalModel(modelDir: Self.q4)
        let state = QwenCPUModel.DecodeState()
        _ = try gpu.step(Self.prompt, state: state)
        let before = try gpu.snapshot(of: state)

        // Checks run: once in step before the token, once before layer 0,
        // then after every layer's command buffer. The third is after layer 0.
        var calls = 0
        #expect(throws: GenerationInterruption.self) {
            _ = try gpu.step([7], state: state, shouldCancel: { calls += 1; return calls == 3 })
        }
        #expect(state.tainted, "cancelling after the first command buffer did not taint the state")
        #expect(state.position == Self.prompt.count)
        #expect(throws: ConversationStateError.inconsistentState("step threw mid-token")) {
            _ = try gpu.snapshot(of: state)
        }
        let other = try QwenMetalModel(modelDir: Self.q4)
        #expect(throws: QwenMetalModel.RuntimeError.self) {
            _ = try other.step([7], state: state)
        }

        let clean = QwenCPUModel.DecodeState()
        _ = try gpu.step(Self.prompt, state: clean)
        var n = 0
        #expect(throws: GenerationInterruption.self) {
            _ = try gpu.step([7], state: clean, shouldCancel: { n += 1; return n == 1 })
        }
        #expect(!clean.tainted)
        #expect(try gpu.snapshot(of: clean) == before, "a cancellation before any layer changed the state")
    }

    /// A live state whose fields do not fit its position is refused by the
    /// writer with the shape problem a reader would otherwise call corrupt.
    @Test func refusesMisShapedLiveState() throws {
        let cpu = try QwenCPUModel(modelDir: Self.q4)
        let state = QwenCPUModel.DecodeState()
        _ = try cpu.step(Self.prompt, state: state)
        let attention = try #require((0..<cpu.config.numHiddenLayers).first { !cpu.config.isLinearLayer($0) })
        let rows = state.kv[attention]!.k.count
        state.kv[attention]!.k.removeLast()
        #expect(throws: ConversationStateError.inconsistentState(
            "layer \(attention) k holds \(rows - 1) floats, geometry says \(rows)")
        ) {
            _ = try cpu.snapshot(of: state)
        }
        state.kv[attention]!.k.append(0)
        #expect(try cpu.snapshot(of: state).count > 0, "repaired state is writable again")
        state.deltaState.removeValue(forKey: 0)
        #expect(throws: ConversationStateError.inconsistentState("layer 0 missing deltaState")) {
            _ = try cpu.snapshot(of: state)
        }
    }
}
