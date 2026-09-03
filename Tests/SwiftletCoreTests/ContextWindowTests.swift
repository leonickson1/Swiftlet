import Foundation
import Testing
@testable import SwiftletCore

@Suite struct ContextWindowTests {
    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    @Test func clampsGenerationToRemainingCapacity() throws {
        let window = try ContextWindow(maximumTokens: 100)

        #expect(try window.admittedMaxNew(
            processedTokens: 0, incomingTokens: 90, requestedMaxNew: 20
        ) == 10)
        #expect(try window.admittedMaxNew(
            processedTokens: 40, incomingTokens: 20, requestedMaxNew: 12
        ) == 12)
    }

    @Test func rejectsPromptWithNoGenerationRoom() throws {
        let window = try ContextWindow(maximumTokens: 100)

        #expect(throws: ContextWindowError.self) {
            _ = try window.admittedMaxNew(
                processedTokens: 40, incomingTokens: 60, requestedMaxNew: 1
            )
        }
        // A caller explicitly requesting prefill only may consume the window.
        #expect(try window.admittedMaxNew(
            processedTokens: 40, incomingTokens: 60, requestedMaxNew: 0
        ) == 0)
    }

    @Test func rejectsOversizedStepBeforeOverflow() throws {
        let window = try ContextWindow(maximumTokens: .max)

        #expect(throws: ContextWindowError.self) {
            try window.validateStep(processedTokens: Int.max - 1, incomingTokens: 2)
        }
        #expect(throws: ContextWindowError.self) {
            _ = try window.admittedMaxNew(
                processedTokens: 0, incomingTokens: 1, requestedMaxNew: -1
            )
        }
    }

    @Test func rejectsEmptyModelStep() throws {
        let window = try ContextWindow(maximumTokens: 100)
        #expect(throws: ContextWindowError.self) {
            try window.validateStep(processedTokens: 0, incomingTokens: 0)
        }
    }

    @Test func realModelStepsRejectBeforeStateMutation() throws {
        let modelDir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let oversized = [Int](repeating: 1, count: 513)

        let cpu = try QwenCPUModel(modelDir: modelDir)
        #expect(cpu.config.maxPositionEmbeddings == 512)
        let cpuState = QwenCPUModel.DecodeState()
        #expect(throws: ContextWindowError.self) {
            _ = try cpu.step(oversized, state: cpuState)
        }
        #expect(cpuState.position == 0)

        let metal = try QwenMetalModel(modelDir: modelDir)
        let metalState = QwenCPUModel.DecodeState()
        #expect(throws: ContextWindowError.self) {
            _ = try metal.step(oversized, state: metalState)
        }
        #expect(metalState.position == 0)
        #expect(metal.lastStepMetrics.tokensProcessed == 0)
        #expect(!metal.lastStepMetrics.completedWithoutThrow)
    }

    /// Rejection on already-primed state: one accepted token first, so the
    /// metrics assertions cannot be satisfied by a step that never ran.
    @Test func primedStateRejectsExactOverflowWithoutMutation() throws {
        let modelDir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let fillsWindow = [Int](repeating: 1, count: 512)

        let cpu = try QwenCPUModel(modelDir: modelDir)
        let cpuState = QwenCPUModel.DecodeState()
        _ = try cpu.step([1], state: cpuState)
        #expect(cpuState.position == 1)
        #expect(throws: ContextWindowError.self) {
            _ = try cpu.step(fillsWindow, state: cpuState)
        }
        #expect(cpuState.position == 1)

        let metal = try QwenMetalModel(modelDir: modelDir)
        let metalState = QwenCPUModel.DecodeState()
        _ = try metal.step([1], state: metalState)
        #expect(metalState.position == 1)
        #expect(metal.lastStepMetrics.tokensProcessed == 1)
        #expect(metal.lastStepMetrics.completedWithoutThrow)
        #expect(throws: ContextWindowError.self) {
            _ = try metal.step(fillsWindow, state: metalState)
        }
        #expect(metalState.position == 1)
        #expect(metal.lastStepMetrics.tokensProcessed == 0)
        #expect(!metal.lastStepMetrics.completedWithoutThrow)
    }

    /// A model that reports a smaller effective capacity than its trained
    /// maximum. `step` records calls and advances the position like the real
    /// implementations; logits are all zero so greedy decode never hits EOS.
    final class FixedCapacityModel: InferenceModel {
        let config: QwenConfig
        let modelDir: URL
        let contextCapacity: Int
        private(set) var steps: [[Int]] = []

        init(modelDir: URL, contextCapacity: Int) throws {
            self.modelDir = modelDir
            self.contextCapacity = contextCapacity
            config = try QwenConfig(url: modelDir.appendingPathComponent("config.json"))
        }

        func step(_ tokens: [Int], state: QwenCPUModel.DecodeState) throws -> [Float] {
            steps.append(tokens)
            state.position += tokens.count
            return [Float](repeating: 0, count: config.vocabSize)
        }
    }

    @Test func generationHonorsImplementationCapacityOverride() throws {
        let modelDir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")

        let small = try FixedCapacityModel(modelDir: modelDir, contextCapacity: 6)
        #expect(small.config.maxPositionEmbeddings == 512)
        let clamped = try TextGenerator(model: small).generate(promptIds: [1, 2, 3], maxNew: 50) { _ in true }
        #expect(clamped.generatedTokens == 3)
        #expect(small.steps.count == 4)

        let trained = try FixedCapacityModel(modelDir: modelDir, contextCapacity: 512)
        let full = try TextGenerator(model: trained).generate(promptIds: [1, 2, 3], maxNew: 50) { _ in true }
        #expect(full.generatedTokens == 50)
    }
}
