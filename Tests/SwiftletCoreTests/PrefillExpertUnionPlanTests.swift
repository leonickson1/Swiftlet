import Foundation
import Testing
@testable import SwiftletCore

/// S1b-a: the expert-union plan is the executable oracle for layer-major
/// prefill. Pure-logic properties first (union correctness, ordering,
/// determinism, bounds), then equivalence against the actual per-token Metal
/// routing on the tiny fixtures.
@Suite struct PrefillExpertUnionPlanTests {
    @Test func unionsExpertsPerLayer() throws {
        let plan = try PrefillExpertUnionPlan(
            tokenRoutes: [
                [[3, 1], [0, 2]],
                [[1, 5], [2, 0]],
                [[3, 1], [7, 4]],
            ],
            expertCount: 8
        )
        #expect(plan.tokenCount == 3)
        #expect(plan.layers.map(\.layerIndex) == [0, 1])
        #expect(plan.layers[0].experts == [1, 3, 5])
        #expect(plan.layers[1].experts == [0, 2, 4, 7])
    }

    @Test func planIsDeterministicAndTokenOrderInvariant() throws {
        let routes: [[[Int]]] = [
            [[2, 0], [1, 3]],
            [[0, 4], [3, 5]],
        ]
        let plan = try PrefillExpertUnionPlan(tokenRoutes: routes, expertCount: 8)
        let again = try PrefillExpertUnionPlan(tokenRoutes: routes, expertCount: 8)
        #expect(plan == again)
        // The union cannot depend on token order or per-token pick order.
        let shuffled = try PrefillExpertUnionPlan(
            tokenRoutes: [[[4, 0], [5, 3]], [[0, 2], [3, 1]]], expertCount: 8
        )
        #expect(plan.layers == shuffled.layers)
    }

    @Test func replayVisitsLayersInAscendingOrder() throws {
        let plan = try PrefillExpertUnionPlan(
            tokenRoutes: [[[1], [2], [0]]], expertCount: 4
        )
        var visited: [(Int, [Int])] = []
        plan.replayLayerMajor { visited.append(($0, $1)) }
        #expect(visited.map(\.0) == [0, 1, 2])
        #expect(visited.map(\.1) == [[1], [2], [0]])
    }

    @Test func malformedRoutesAreRejected() {
        #expect(throws: PrefillExpertUnionPlan.BuildError.emptyPrompt) {
            try PrefillExpertUnionPlan(tokenRoutes: [], expertCount: 8)
        }
        #expect(throws: PrefillExpertUnionPlan.BuildError.inconsistentLayerCount(
            token: 1, expected: 2, actual: 1
        )) {
            try PrefillExpertUnionPlan(tokenRoutes: [[[0], [1]], [[0]]], expertCount: 8)
        }
        #expect(throws: PrefillExpertUnionPlan.BuildError.emptyRoute(token: 0, layer: 1)) {
            try PrefillExpertUnionPlan(tokenRoutes: [[[0], []]], expertCount: 8)
        }
        #expect(throws: PrefillExpertUnionPlan.BuildError.expertOutOfRange(
            token: 0, layer: 0, expert: 8
        )) {
            try PrefillExpertUnionPlan(tokenRoutes: [[[8]]], expertCount: 8)
        }
        #expect(throws: PrefillExpertUnionPlan.BuildError.expertOutOfRange(
            token: 0, layer: 0, expert: -1
        )) {
            try PrefillExpertUnionPlan(tokenRoutes: [[[-1]]], expertCount: 8)
        }
        #expect(throws: PrefillExpertUnionPlan.BuildError.duplicateExpert(
            token: 0, layer: 0, expert: 3
        )) {
            try PrefillExpertUnionPlan(tokenRoutes: [[[3, 3]]], expertCount: 8)
        }
    }

    @Test func unionSizeIsBounded() throws {
        let plan = try PrefillExpertUnionPlan(
            tokenRoutes: [[[0, 1]], [[2, 3]], [[0, 2]]], expertCount: 4
        )
        for layer in plan.layers {
            #expect(layer.experts.count <= 4, "union exceeds the expert count")
            #expect(layer.experts.count <= 3 * 2, "union exceeds tokens * top-k")
            #expect(layer.experts == Array(Set(layer.experts)).sorted(),
                    "experts not unique/ascending")
        }
    }

    // MARK: - Equivalence against the current Metal token-by-token path

    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    /// Regroups the flat (layer, experts) router observations one step call
    /// produced into token-major routes, checking the observation shape.
    static func tokenMajor(
        _ flat: [(layer: Int, experts: [Int])], tokens: Int, layerCount: Int
    ) -> [[[Int]]] {
        guard flat.count == tokens * layerCount else {
            Issue.record("router observations \(flat.count) != \(tokens * layerCount)")
            return []
        }
        var result: [[[Int]]] = []
        for t in 0..<tokens {
            var perLayer: [[Int]] = []
            for l in 0..<layerCount {
                let entry = flat[t * layerCount + l]
                #expect(entry.layer == l, "layer order diverged at token \(t)")
                perLayer.append(entry.experts)
            }
            result.append(perLayer)
        }
        return result
    }

    /// The plan replayed layer-by-layer must touch exactly the expert sets
    /// the current token-by-token path touches for the same prompt.
    static func expectPlanMatchesRouting(_ modelName: String) throws {
        let dir = fixturesDir.appendingPathComponent(modelName)
        let tokens = [1, 5, 9, 42, 7]

        // Decode-style loop: one step call per token.
        let loopModel = try QwenMetalModel(modelDir: dir)
        let layerCount = loopModel.config.numHiddenLayers
        var loopFlat: [(layer: Int, experts: [Int])] = []
        loopModel.routedExpertObserver = { loopFlat.append(($0, $1)) }
        let loopState = loopModel.makeQwenContext()
        for t in tokens { _ = try loopModel.step([t], context: loopState) }
        loopModel.routedExpertObserver = nil

        // S1a prefill-style path: one multi-token prompt call, pinned to the
        // token-major schedule this equivalence was defined against. The S1b
        // layer-major schedule's route/union equivalence is asserted in
        // LayerMajorPrefillTests.
        let promptModel = try QwenMetalModel(modelDir: dir)
        promptModel.prefillMode = .tokenMajor
        var promptFlat: [(layer: Int, experts: [Int])] = []
        promptModel.routedExpertObserver = { promptFlat.append(($0, $1)) }
        let promptState = promptModel.makeQwenContext()
        _ = try promptModel.step(tokens, context: promptState)
        promptModel.routedExpertObserver = nil

        let loopRoutes = tokenMajor(loopFlat, tokens: tokens.count, layerCount: layerCount)
        let promptRoutes = tokenMajor(promptFlat, tokens: tokens.count, layerCount: layerCount)
        #expect(loopRoutes == promptRoutes,
                "\(modelName): routing diverged between step loop and prompt call")
        for perToken in loopRoutes {
            for route in perToken {
                #expect(route.count == loopModel.config.numExpertsPerTok,
                        "\(modelName): route width diverged from top-k")
            }
        }

        let plan = try PrefillExpertUnionPlan(
            tokenRoutes: loopRoutes, expertCount: loopModel.config.numExperts
        )
        #expect(plan.tokenCount == tokens.count)
        #expect(plan.layers.count == layerCount)

        // Replay layer-by-layer and compare the touched expert sets against
        // the token-by-token path, layer for layer.
        var replayed: [Int: Set<Int>] = [:]
        plan.replayLayerMajor { layer, experts in
            replayed[layer] = Set(experts)
        }
        var touched: [Int: Set<Int>] = [:]
        for perToken in loopRoutes {
            for (layer, experts) in perToken.enumerated() {
                touched[layer, default: []].formUnion(experts)
            }
        }
        #expect(replayed == touched,
                "\(modelName): plan does not touch the token-by-token expert set")
        for layer in plan.layers {
            #expect(layer.experts.count >= loopModel.config.numExpertsPerTok,
                    "\(modelName): union smaller than a single token's route")
        }
    }

    @Test func planMatchesMetalRoutingOnQuantizedTiny() throws {
        try Self.expectPlanMatchesRouting("tiny-model-q4")
    }

    @Test func planMatchesMetalRoutingOnQwen35Tiny() throws {
        try Self.expectPlanMatchesRouting("tiny-model-q35")
    }
}
