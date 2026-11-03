import Foundation
import Testing
@testable import SwiftletCore

/// The Metal fast path keeps the bound state's conv history and delta
/// recurrence in per-layer GPU buffers. Stepping a different `DecodeState`
/// must capture the outgoing state's recurrence and load the incoming one's
/// (plus its KV rows, which the other state overwrote at the same
/// positions). Before `bindState` the buffers were only re-zeroed at
/// position 0, so a second live state — or a state at position > 0 that had
/// never been bound — continued from the wrong recurrence.
@Suite struct MetalStateSwitchTests {
    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")
    static let q4 = fixturesDir.appendingPathComponent("tiny-model-q4")

    static let a = [1, 5, 9, 42, 7, 99]
    static let b = [3, 17, 64, 2, 11, 5]

    /// Multi-token prefill of the first three tokens, then single-token
    /// decode: both step shapes. Returns one logits vector per step.
    static func isolated(_ model: any InferenceModel, _ seq: [Int]) throws
        -> (logits: [[Float]], state: QwenCPUModel.DecodeState) {
        let state = QwenCPUModel.DecodeState()
        var out = [try model.step(Array(seq[0..<3]), state: state)]
        for t in seq[3...] { out.append(try model.step([t], state: state)) }
        return (out, state)
    }

    /// One model, two live states, stepped alternately, must produce the
    /// same logits and KV rows as each sequence run alone on that model.
    @Test func interleavedStatesMatchIsolatedRuns() throws {
        let a = Self.a, b = Self.b

        func interleaved(_ model: any InferenceModel) throws
            -> (a: [[Float]], b: [[Float]], stateA: QwenCPUModel.DecodeState, stateB: QwenCPUModel.DecodeState) {
            let stateA = QwenCPUModel.DecodeState()
            let stateB = QwenCPUModel.DecodeState()
            var outA = [try model.step(Array(a[0..<3]), state: stateA)]
            var outB = [try model.step(Array(b[0..<3]), state: stateB)]
            for i in 3..<a.count {
                outA.append(try model.step([a[i]], state: stateA))
                outB.append(try model.step([b[i]], state: stateB))
            }
            return (outA, outB, stateA, stateB)
        }

        func check(_ model: any InferenceModel, label: String) throws {
            let refA = try Self.isolated(model, a)
            let refB = try Self.isolated(model, b)
            let mixed = try interleaved(model)
            #expect(mixed.a == refA.logits, "\(label): sequence A diverges when interleaved")
            #expect(mixed.b == refB.logits, "\(label): sequence B diverges when interleaved")
            #expect(mixed.stateA.position == a.count && mixed.stateB.position == b.count)
            for (layer, ref) in refA.state.kv {
                let got = try #require(mixed.stateA.kv[layer], "\(label): A lost KV layer \(layer)")
                #expect(got.k == ref.k && got.v == ref.v, "\(label): A KV layer \(layer) diverged")
            }
            for (layer, ref) in refB.state.kv {
                let got = try #require(mixed.stateB.kv[layer], "\(label): B lost KV layer \(layer)")
                #expect(got.k == ref.k && got.v == ref.v, "\(label): B KV layer \(layer) diverged")
            }
        }

        let cpu = try QwenCPUModel(modelDir: Self.q4)
        cpu.retainAllLayers = true
        try check(cpu, label: "CPU")
        try check(try QwenMetalModel(modelDir: Self.q4), label: "Metal")
    }

    /// A state at position > 0 that this instance has never bound (here:
    /// stepped on another Metal instance, then captured there by a switch)
    /// must load into the buffers of an instance that is midway through a
    /// different sequence, and the displaced sequence must resume bitwise.
    @Test func unboundStateLoadsIntoBusyInstance() throws {
        let a = Self.a, b = Self.b
        let reference = try QwenMetalModel(modelDir: Self.q4)
        let refA = try Self.isolated(reference, a).logits
        let refB = try Self.isolated(reference, b).logits

        // b[0..<5] on `other`; the switch to a throwaway state captures it.
        let other = try QwenMetalModel(modelDir: Self.q4)
        let stateB = QwenCPUModel.DecodeState()
        var outB = [try other.step(Array(b[0..<3]), state: stateB)]
        for t in b[3..<5] { outB.append(try other.step([t], state: stateB)) }
        #expect(other.boundState === stateB && stateB.boundOn === other)
        _ = try other.step([1], state: QwenCPUModel.DecodeState())
        #expect(other.boundState !== stateB && stateB.boundOn == nil, "switch did not release the state")
        for li in 0..<other.config.numHiddenLayers where other.config.isLinearLayer(li) {
            #expect(stateB.convTail[li] != nil && stateB.deltaState[li] != nil,
                    "layer \(li): capture left no recurrence on the released state")
        }

        // a[0..<5] on `busy`, then b's last token from the foreign state.
        let busy = try QwenMetalModel(modelDir: Self.q4)
        let stateA = QwenCPUModel.DecodeState()
        var outA = [try busy.step(Array(a[0..<3]), state: stateA)]
        for t in a[3..<5] { outA.append(try busy.step([t], state: stateA)) }
        outB.append(try busy.step([b[5]], state: stateB))
        #expect(busy.boundState === stateB && stateB.boundOn === busy && stateA.boundOn == nil)
        outA.append(try busy.step([a[5]], state: stateA))
        #expect(busy.boundState === stateA && stateA.boundOn === busy && stateB.boundOn == nil)

        #expect(outB == refB, "Metal: sequence B diverges after loading into a busy instance")
        #expect(outA == refA, "Metal: sequence A diverges after being displaced and resumed")
    }

    /// A state the GPU buffers cannot hold is refused before anything is
    /// captured or written: the bound state stays bound and continues
    /// bitwise.
    @Test func misShapedStateIsRefusedWithoutDisturbingTheBoundOne() throws {
        let gpu = try QwenMetalModel(modelDir: Self.q4)
        let refA = try Self.isolated(try QwenMetalModel(modelDir: Self.q4), Self.a).logits
        let stateA = QwenCPUModel.DecodeState()
        var outA = [try gpu.step(Array(Self.a[0..<3]), state: stateA)]
        for t in Self.a[3..<5] { outA.append(try gpu.step([t], state: stateA)) }

        // Position > 0 with no fields at all: nothing to load.
        let bogus = QwenCPUModel.DecodeState()
        bogus.position = 2
        #expect(throws: QwenMetalModel.RuntimeError.self) {
            _ = try gpu.step([7], state: bogus)
        }
        #expect(gpu.boundState === stateA && stateA.boundOn === gpu, "refusal disturbed the binding")
        outA.append(try gpu.step([Self.a[5]], state: stateA))
        #expect(outA == refA, "Metal: bound state diverged after a refused switch")
    }

    /// When the bound instance dies, the state gets its recurrence back and
    /// is no longer marked as bound anywhere, so another instance can load
    /// it and continue bitwise.
    @Test func deinitHandsTheBoundStateBack() throws {
        let refB = try Self.isolated(try QwenMetalModel(modelDir: Self.q4), Self.b).logits
        let stateB = QwenCPUModel.DecodeState()
        var outB: [[Float]] = []
        do {
            let dying = try QwenMetalModel(modelDir: Self.q4)
            outB.append(try dying.step(Array(Self.b[0..<3]), state: stateB))
            for t in Self.b[3..<5] { outB.append(try dying.step([t], state: stateB)) }
            #expect(stateB.boundOn === dying)
        }
        #expect(stateB.boundOn == nil, "deinit left the state marked as bound")
        let successor = try QwenMetalModel(modelDir: Self.q4)
        for li in 0..<successor.config.numHiddenLayers where successor.config.isLinearLayer(li) {
            #expect(stateB.convTail[li] != nil && stateB.deltaState[li] != nil,
                    "layer \(li): deinit left no recurrence on the state")
        }
        outB.append(try successor.step([Self.b[5]], state: stateB))
        #expect(outB == refB, "Metal: state handed back by deinit diverges on a successor instance")
    }
}
