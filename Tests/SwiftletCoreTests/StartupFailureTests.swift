import Foundation
import Hub
import Testing
import Tokenizers
@testable import SwiftletCore

/// Every error class a model can throw on the way to a ready session maps
/// onto one `StartupFailure` kind, and the kinds carry distinct exit codes.
@Suite struct StartupFailureTests {
    private static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    private func kind(_ error: Swift.Error) -> String {
        StartupFailure.classify(error).kindName
    }

    @Test func configErrorsAreConfig() {
        #expect(kind(QwenConfig.Error.unsupportedModelType("glm5_next")) == "config")
        #expect(kind(QwenConfig.Error.missingModelType) == "config")
        #expect(kind(QwenConfig.Error.missingField("hidden_size")) == "config")
        #expect(kind(QwenConfig.Error.invalidField("num_experts must be > 0")) == "config")
        #expect(kind(Checkpoint.Error.malformedConfig(path: "/m/config.json", reason: "not an object")) == "config")
        #expect(kind(Checkpoint.Error.unsupportedQuantMode(mode: "mxfp4", module: "quantization")) == "config")
        #expect(kind(Checkpoint.Error.unsupportedBits(3)) == "config")
    }

    @Test func containerErrorsAreContainer() {
        #expect(kind(Checkpoint.Error.missingConfig("/m")) == "container")
        #expect(kind(Checkpoint.Error.missingTensor("model.norm.weight")) == "container")
        #expect(kind(Checkpoint.Error.badShape("corrupt container: empty expert layout")) == "container")
        #expect(kind(Qpack.Error.notAContainer("/m")) == "container")
        #expect(kind(Qpack.Error.missingFile("packed_experts/layer_02.bin")) == "container")
        #expect(kind(Qpack.Error.sizeMismatch(path: "model.safetensors", expected: 2, actual: 1)) == "container")
        #expect(kind(Qpack.Error.layoutMismatch("layer_01.bin")) == "container")
        #expect(kind(SafetensorsFile.Error.malformedHeader) == "container")
        #expect(kind(SafetensorsFile.Error.missingTensor("x")) == "container")
        #expect(kind(SafetensorsFile.Error.unsupportedDtype("I2", tensor: "x")) == "container")
        #expect(kind(MetalShardStore.Error.mapFailed("model.safetensors")) == "container")
        #expect(kind(MetalShardStore.Error.badAlignment("model.safetensors")) == "container")
        #expect(kind(Hub.HubClientError.configurationMissing("tokenizer.json")) == "container")
        #expect(kind(TokenizerError.missingVocab) == "container")
        #expect(kind(CocoaError(.fileNoSuchFile)) == "container")
        #expect(kind(CocoaError(.fileReadNoSuchFile)) == "container")
        #expect(kind(POSIXError(.ENOENT)) == "container")
    }

    @Test func resourceErrorsAreResource() {
        #expect(kind(ExpertCache.Error.budgetTooSmall(slots: 0, required: 16)) == "resource")
        #expect(kind(QwenMetalModel.RuntimeError.kvCacheAllocationFailed(layer: 3, rows: 256)) == "resource")
        #expect(kind(POSIXError(.ENOMEM)) == "resource")
    }

    @Test func backendErrorsAreBackend() {
        struct Unnamed: Swift.Error {}
        #expect(kind(MetalEngine.Error.noDevice) == "backend")
        #expect(kind(MetalEngine.Error.kernelMissing("gemv_q4")) == "backend")
        #expect(kind(Unnamed()) == "backend")
        #expect(kind(CocoaError(.featureUnsupported)) == "backend")
    }

    @Test func classifiedFailurePassesThroughUnchanged() {
        let failure = StartupFailure.resource("cannot listen on 127.0.0.1:8080: in use")
        #expect(StartupFailure.classify(failure) == failure)
    }

    @Test func reasonIsTheUnderlyingDescription() {
        let error = QwenConfig.Error.unsupportedModelType("glm5_next")
        let failure = StartupFailure.classify(error)
        #expect(failure.reason == String(describing: error))
        #expect(failure.description == "config: " + String(describing: error))
        // LocalizedError text wins where a type supplies it.
        let hub = Hub.HubClientError.configurationMissing("tokenizer.json")
        #expect(StartupFailure.classify(hub).reason == hub.errorDescription)
    }

    @Test func exitCodesAreDistinctAndNeverUsageOrGeneric() {
        let kinds: [StartupFailure] = [.container(""), .config(""), .resource(""), .backend("")]
        let codes = kinds.map(\.exitCode)
        #expect(Set(codes).count == kinds.count)
        for code in codes {
            #expect(code != 0)
            #expect(code != 1)
            #expect(code != 2)
        }
        #expect(StartupFailure.container("").exitCode == 3)
        #expect(StartupFailure.config("").exitCode == 4)
        #expect(StartupFailure.resource("").exitCode == 5)
        #expect(StartupFailure.backend("").exitCode == 6)
    }

    // MARK: - Through the real session initializer (no GPU needed: both
    // failures are raised before any Metal object exists).

    @Test func missingDirectoryOpensAsContainerFailure() async {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("swiftlet-missing-\(UUID().uuidString)")
        do {
            _ = try await SwiftletSession(modelDir: missing)
            Issue.record("a missing model directory opened")
        } catch {
            #expect(kind(error) == "container", "\(error)")
        }
    }

    @Test func refusedModelTypeOpensAsConfigFailure() async throws {
        let copy = FileManager.default.temporaryDirectory
            .appendingPathComponent("swiftlet-model-type-\(UUID().uuidString)")
        try FileManager.default.copyItem(
            at: Self.fixturesDir.appendingPathComponent("tiny-model"), to: copy
        )
        defer { try? FileManager.default.removeItem(at: copy) }
        let configURL = copy.appendingPathComponent("config.json")
        var config = try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any]
        )
        config["model_type"] = "glm5_next"
        try JSONSerialization.data(withJSONObject: config).write(to: configURL)

        do {
            _ = try await SwiftletSession(modelDir: copy)
            Issue.record("a checkpoint naming model_type glm5_next opened")
        } catch {
            #expect(kind(error) == "config", "\(error)")
            #expect(StartupFailure.classify(error).reason.contains("glm5_next"))
        }
    }
}
