import Foundation

/// S1b-a: executable planning oracle for layer-major prefill.
///
/// Given the per-token router outcomes a prompt produced, the plan holds, for
/// every layer, the union of experts any prompt token routed to — the exact
/// expert set a layer-major, chunked prefill must have resident while it
/// sweeps that layer across all positions. This is pure CPU logic: it changes
/// no existing schedule and is consumed on the CPU by tests today and by the
/// S1b layer-major prefill path later.
public struct PrefillExpertUnionPlan: Equatable, Sendable {
    /// One layer's expert requirement.
    public struct LayerPlan: Equatable, Sendable {
        public let layerIndex: Int
        /// Unique routed experts, ascending. Deterministic for a given set of
        /// router outcomes regardless of token order or per-token pick order.
        public let experts: [Int]
    }

    public enum BuildError: Swift.Error, Equatable {
        /// A layer-major prefill over zero tokens is meaningless.
        case emptyPrompt
        /// Every token must report a route for the same set of layers.
        case inconsistentLayerCount(token: Int, expected: Int, actual: Int)
        /// The router always selects at least one expert per token and layer.
        case emptyRoute(token: Int, layer: Int)
        /// Expert index outside 0..<expertCount.
        case expertOutOfRange(token: Int, layer: Int, expert: Int)
        /// The router never picks the same expert twice for one token.
        case duplicateExpert(token: Int, layer: Int, expert: Int)
    }

    public let tokenCount: Int
    /// Ascending by layerIndex, one entry per layer.
    public let layers: [LayerPlan]

    /// Builds the plan from token-major router outcomes:
    /// `tokenRoutes[token][layer]` lists the experts that token routed to at
    /// that layer, in any order. Malformed input is rejected, not repaired —
    /// an oracle that silently fixes its input proves nothing.
    public init(tokenRoutes: [[[Int]]], expertCount: Int) throws {
        guard let first = tokenRoutes.first else { throw BuildError.emptyPrompt }
        let layerCount = first.count
        var unions = [Set<Int>](repeating: [], count: layerCount)
        for (token, perLayer) in tokenRoutes.enumerated() {
            guard perLayer.count == layerCount else {
                throw BuildError.inconsistentLayerCount(
                    token: token, expected: layerCount, actual: perLayer.count
                )
            }
            for (layer, route) in perLayer.enumerated() {
                guard !route.isEmpty else {
                    throw BuildError.emptyRoute(token: token, layer: layer)
                }
                var seen = Set<Int>()
                for expert in route {
                    guard expert >= 0, expert < expertCount else {
                        throw BuildError.expertOutOfRange(
                            token: token, layer: layer, expert: expert
                        )
                    }
                    guard seen.insert(expert).inserted else {
                        throw BuildError.duplicateExpert(
                            token: token, layer: layer, expert: expert
                        )
                    }
                    unions[layer].insert(expert)
                }
            }
        }
        tokenCount = tokenRoutes.count
        layers = unions.enumerated().map {
            LayerPlan(layerIndex: $0.offset, experts: $0.element.sorted())
        }
    }

    /// Replays the plan in the order a layer-major prefill would consume it:
    /// ascending layers, each with its ascending expert union.
    public func replayLayerMajor(
        _ body: (_ layerIndex: Int, _ experts: [Int]) throws -> Void
    ) rethrows {
        for layer in layers {
            try body(layer.layerIndex, layer.experts)
        }
    }
}
