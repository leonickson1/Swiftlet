import Foundation
import Testing
@testable import SwiftletCore

/// Batching independent GEMVs into gemv_affine_fast_multi dispatches must
/// compute exactly what one dispatch per GEMV did.
@Suite struct BatchedGemvTests {
    static func repacked(_ modelName: String) throws -> URL {
        let src = MetalModelTests.fixturesDir.appendingPathComponent(modelName)
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("\(modelName)-batched-\(UUID().uuidString).qpack")
        var repacker = QpackRepacker(checkpointDir: src, outputDir: out)
        repacker.log = { _ in }
        try repacker.repack()
        return out
    }

    /// Logits bitwise equal with batching on and off, from a .qpack container
    /// (experts in cache slots) and from the raw checkpoint (stacked experts).
    @Test(arguments: [("tiny-model-q4", true), ("tiny-model-q4", false),
                      ("tiny-model-q35", true), ("tiny-model-q35", false)])
    func batchingNeverChangesTheLogits(modelName: String, qpack: Bool) throws {
        let dir = qpack ? try Self.repacked(modelName)
                        : MetalModelTests.fixturesDir.appendingPathComponent(modelName)
        defer { if qpack { try? FileManager.default.removeItem(at: dir) } }
        let batched = try QwenMetalModel(modelDir: dir, cacheBudgetGB: 0.05)
        let single = try QwenMetalModel(modelDir: dir, cacheBudgetGB: 0.05)
        single.batchGemvs = false
        let a = QwenCPUModel.DecodeState(), b = QwenCPUModel.DecodeState()
        for t in [1, 5, 9, 42, 7, 99, 3, 17] {
            let x = try batched.step([t], state: a)
            let y = try single.step([t], state: b)
            #expect(x == y, "\(modelName) qpack=\(qpack): logits diverged at token \(t)")
        }
        // tiny-model-q4 from a .qpack (8 layers, top-2, 6 DeltaNet + 2
        // attention): per layer the 2K + 3 first-stage GEMVs become 1 (-6)
        // and the K + 1 down projections become 1 (-2); q/k/v become 1 (-2
        // per attention layer); DeltaNet's two input projections become 1 (-1
        // per DeltaNet layer): 8 * 8 + 2 * 2 + 6 = 74. The raw checkpoint's
        // weights sit at unaligned file offsets, so every GEMV keeps its own
        // dispatch (as the per-GEMV fast path never applied there either);
        // tiny-model-q35 is unquantized.
        let saved = single.lastStepMetrics.computeDispatchesEncoded
            - batched.lastStepMetrics.computeDispatchesEncoded
        #expect(saved == (modelName == "tiny-model-q4" && qpack ? 74 : 0))
    }
}
