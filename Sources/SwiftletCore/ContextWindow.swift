import Foundation

/// Stable failures for context-window admission. Counts are kept separate in
/// the error payload so reporting an oversized request never has to perform an
/// overflowing addition.
public enum ContextWindowError: Swift.Error, Equatable, LocalizedError, Sendable {
    case invalidMaximum(Int)
    case invalidTokenCount(name: String, value: Int)
    case emptyInput
    case capacityExceeded(processed: Int, incoming: Int, maximum: Int)
    case noGenerationCapacity(processed: Int, incoming: Int, maximum: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidMaximum(let maximum):
            return "context window must be positive (got \(maximum))"
        case .invalidTokenCount(let name, let value):
            return "\(name) must be nonnegative (got \(value))"
        case .emptyInput:
            return "at least one input token is required"
        case .capacityExceeded(let processed, let incoming, let maximum):
            return "context window exceeded: \(processed) processed + \(incoming) incoming tokens, maximum \(maximum)"
        case .noGenerationCapacity(let processed, let incoming, let maximum):
            return "no context capacity remains for generation after \(processed) processed + \(incoming) incoming tokens (maximum \(maximum))"
        }
    }
}

/// Overflow-safe admission against the effective model context capacity.
public struct ContextWindow: Sendable {
    public let maximumTokens: Int

    public init(maximumTokens: Int) throws {
        guard maximumTokens > 0 else { throw ContextWindowError.invalidMaximum(maximumTokens) }
        self.maximumTokens = maximumTokens
    }

    /// Validates one model step before it mutates recurrent or KV state.
    public func validateStep(processedTokens: Int, incomingTokens: Int) throws {
        guard processedTokens >= 0 else {
            throw ContextWindowError.invalidTokenCount(
                name: "processedTokens", value: processedTokens
            )
        }
        guard incomingTokens >= 0 else {
            throw ContextWindowError.invalidTokenCount(
                name: "incomingTokens", value: incomingTokens
            )
        }
        guard incomingTokens > 0 else { throw ContextWindowError.emptyInput }
        guard processedTokens <= maximumTokens,
              incomingTokens <= maximumTokens - processedTokens else {
            throw ContextWindowError.capacityExceeded(
                processed: processedTokens, incoming: incomingTokens, maximum: maximumTokens
            )
        }
    }

    /// Returns the requested output count clamped to the capacity remaining
    /// after the incoming prompt. A positive request with no room is rejected
    /// before prefill instead of failing partway through generation.
    public func admittedMaxNew(
        processedTokens: Int, incomingTokens: Int, requestedMaxNew: Int
    ) throws -> Int {
        guard requestedMaxNew >= 0 else {
            throw ContextWindowError.invalidTokenCount(
                name: "requestedMaxNew", value: requestedMaxNew
            )
        }
        try validateStep(processedTokens: processedTokens, incomingTokens: incomingTokens)
        let remaining = maximumTokens - processedTokens - incomingTokens
        guard requestedMaxNew == 0 || remaining > 0 else {
            throw ContextWindowError.noGenerationCapacity(
                processed: processedTokens, incoming: incomingTokens, maximum: maximumTokens
            )
        }
        return min(requestedMaxNew, remaining)
    }
}
