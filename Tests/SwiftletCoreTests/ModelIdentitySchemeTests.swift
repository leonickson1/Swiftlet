import CryptoKit
import Foundation
import Testing
@testable import SwiftletCore

/// Identity scheme 1: a model directory carrying `hashes.json` — the per-file
/// SHA-256 manifest `scripts/verify_container.py --write-hashes` writes and
/// `StreamingInstaller` verifies downloads against,
/// `{ "files": { "<relative path>": "<hex sha256>" } }` — is named by
/// `config.json` and that manifest, not by its shard headers. No fixture
/// ships one, so each test builds its own: a copy of `fixtures/tiny-model-q4`
/// in a temporary directory, hashed the way the script hashes a container, so
/// the digests always match the files they cover.
@Suite struct ModelIdentitySchemeTests {
    static let q4 = ConversationStateTests.q4

    /// A copy of tiny-model-q4 (config.json + model.safetensors) in its own
    /// temporary directory, removed when the value goes away.
    final class ModelCopy {
        let dir: URL

        init() throws {
            dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("swiftlet-scheme1-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            for name in ["config.json", "model.safetensors"] {
                try FileManager.default.copyItem(
                    at: ModelIdentitySchemeTests.q4.appendingPathComponent(name),
                    to: dir.appendingPathComponent(name))
            }
        }

        deinit { try? FileManager.default.removeItem(at: dir) }

        var hashesURL: URL { dir.appendingPathComponent("hashes.json") }

        /// What `verify_container.py --write-hashes` would write here: every
        /// file except dotfiles and hashes.json itself, keyed by relative path.
        func installerManifest() throws -> [String: String] {
            var files: [String: String] = [:]
            for name in try FileManager.default.contentsOfDirectory(atPath: dir.path)
            where !name.hasPrefix(".") && name != "hashes.json" {
                let bytes = try Data(contentsOf: dir.appendingPathComponent(name))
                files[name] = ModelIdentitySchemeTests.sha256Hex(bytes)
            }
            return files
        }

        /// Writes `files` in the installer's shape (pretty-printed, keys sorted,
        /// so the same table always gives the same bytes).
        func writeHashes(_ files: [String: String]) throws {
            let bytes = try JSONSerialization.data(
                withJSONObject: ["files": files], options: [.sortedKeys, .prettyPrinted])
            try writeHashes(bytes)
        }

        func writeHashes(_ bytes: Data) throws { try bytes.write(to: hashesURL) }

        func removeHashes() throws { try FileManager.default.removeItem(at: hashesURL) }

        /// The identity exactly as both engines compute it for their
        /// `stateIdentity`: the checkpoint's directory and shard URLs.
        func identity() throws -> ModelIdentity {
            let ckpt = try Checkpoint(dir: dir)
            return try ModelIdentity.compute(modelDir: dir, shards: ckpt.shardURLs)
        }

        func cpu() throws -> QwenCPUModel {
            let m = try QwenCPUModel(modelDir: dir)
            m.retainAllLayers = true
            return m
        }
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// A snapshot of `model` after the shared prompt (position prompt.count).
    static func snapshotAfterPrompt(_ model: any PersistableInferenceModel) throws -> Data {
        let state = QwenCPUModel.DecodeState()
        _ = try model.step(ConversationStateTests.prompt, state: state)
        return try model.snapshot(of: state)
    }

    /// A directory with hashes.json is scheme 1; the same files without it
    /// are scheme 2 with a different digest; and the scheme-1 digest is
    /// SHA-256 over (label, length, bytes) of config.json then hashes.json —
    /// nothing read from the shard itself.
    @Test func hashesJSONSelectsSchemeOne() throws {
        let copy = try ModelCopy()
        let plain = try copy.identity()
        #expect(plain.scheme == 2)
        let fixture = try Checkpoint(dir: Self.q4)
        #expect(try ModelIdentity.compute(modelDir: Self.q4, shards: fixture.shardURLs) == plain,
                "a copy of the fixture does not share its scheme-2 identity")

        try copy.writeHashes(try copy.installerManifest())
        let hashed = try copy.identity()
        #expect(hashed.scheme == 1)
        #expect(hashed.digest.count == 32)
        #expect(hashed.digest != plain.digest, "hashes.json did not change the digest")
        #expect(try copy.identity() == hashed, "scheme-1 identity is not deterministic")
        #expect(try copy.cpu().stateIdentity == hashed, "the engine's stateIdentity is not compute()'s")

        var expected = SHA256()
        for name in ["config.json", "hashes.json"] {
            let bytes = try Data(contentsOf: copy.dir.appendingPathComponent(name))
            expected.update(data: Data(name.utf8))
            var length = UInt64(bytes.count).littleEndian
            withUnsafeBytes(of: &length) { expected.update(bufferPointer: $0) }
            expected.update(data: bytes)
        }
        #expect(Array(expected.finalize()) == hashed.digest, "scheme-1 digest construction")

        try copy.removeHashes()
        #expect(try copy.identity() == plain, "removing hashes.json does not restore the scheme-2 identity")
    }

    /// A snapshot written under the hashed directory restores into it
    /// bitwise and carries scheme 1 in its header; the same files minus
    /// hashes.json refuse it as modelMismatch (the scheme changed), and the
    /// plain directory's snapshot is refused by the hashed one the same way —
    /// on the CPU engine and on Metal.
    @Test func schemeOneSnapshotRoundTripsAndIsRefusedAcrossSchemes() throws {
        let hashed = try ModelCopy()
        try hashed.writeHashes(try hashed.installerManifest())
        let plain = try ModelCopy()
        let idH = try hashed.identity()
        let idP = try plain.identity()
        #expect(idH.scheme == 1)
        #expect(idP.scheme == 2)

        let model = try hashed.cpu()
        let reference = try ConversationStateTests.uninterrupted(model)
        let (resumed, data) = try ConversationStateTests.interrupted(model) {
            try hashed.cpu() as any PersistableInferenceModel
        }
        #expect(resumed.tokens == reference.tokens, "scheme 1: greedy tokens diverge after restore")
        #expect(resumed.logits == reference.logits, "scheme 1: logits diverge after restore")
        #expect(reference.tokens.count == ConversationStateTests.savedAfter + ConversationStateTests.continued)
        let info = try ConversationState.info(of: data)
        #expect(info.identityScheme == 1)
        #expect(info.identityHex == idH.hex)
        let schemeField = data.subdata(in: 20..<24).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
        #expect(schemeField == 1, "identity scheme at header offset 20")
        #expect(Array(data.subdata(in: 24..<56)) == idH.digest, "identity digest at header offset 24")

        // Same files, no hashes.json: refused, naming both identities.
        let plainModel = try plain.cpu()
        #expect(throws: ConversationStateError.modelMismatch(expected: idP.hex, found: idH.hex)) {
            _ = try plainModel.restoreState(from: data)
        }
        // And the reverse.
        let fromPlain = try Self.snapshotAfterPrompt(plainModel)
        #expect(try ConversationState.info(of: fromPlain).identityScheme == 2)
        #expect(throws: ConversationStateError.modelMismatch(expected: idH.hex, found: idP.hex)) {
            _ = try model.restoreState(from: fromPlain)
        }
        // The Metal engine names the hashed directory the same way.
        let gpu = try QwenMetalModel(modelDir: hashed.dir)
        #expect(try gpu.restoreState(from: data).position == info.position)
        #expect(throws: ConversationStateError.modelMismatch(expected: idH.hex, found: idP.hex)) {
            _ = try gpu.restoreState(from: fromPlain)
        }
    }

    /// The identity is over hashes.json's bytes: one digest edited (the files
    /// on disk unchanged) is another model and refuses the snapshot; the same
    /// table in another encoding is another identity too; the original bytes
    /// name the original model again.
    @Test func editedDigestChangesTheIdentity() throws {
        let copy = try ModelCopy()
        var files = try copy.installerManifest()
        try copy.writeHashes(files)
        let original = try copy.identity()
        #expect(original.scheme == 1)
        let data = try Self.snapshotAfterPrompt(try copy.cpu())
        #expect(try copy.cpu().restoreState(from: data).position == ConversationStateTests.prompt.count)

        // Flip the last hex digit of the shard's digest.
        let shard = try #require(files["model.safetensors"])
        files["model.safetensors"] = String(shard.dropLast()) + (shard.last == "0" ? "1" : "0")
        try copy.writeHashes(files)
        let edited = try copy.identity()
        #expect(edited.scheme == 1)
        #expect(edited.digest != original.digest, "an edited digest did not change the identity")
        #expect(throws: ConversationStateError.modelMismatch(expected: edited.hex, found: original.hex)) {
            _ = try copy.cpu().restoreState(from: data)
        }

        // Bytes, not table: the original table, compact instead of pretty-printed.
        files["model.safetensors"] = shard
        let compactBytes = try JSONSerialization.data(withJSONObject: ["files": files], options: [.sortedKeys])
        try copy.writeHashes(compactBytes)
        let compact = try copy.identity()
        #expect(compact.scheme == 1)
        #expect(compact.digest != original.digest, "re-encoding hashes.json did not change the identity")
        #expect(throws: ConversationStateError.modelMismatch(expected: compact.hex, found: original.hex)) {
            _ = try copy.cpu().restoreState(from: data)
        }

        // The original bytes again.
        try copy.writeHashes(files)
        #expect(try copy.identity() == original)
        #expect(try copy.cpu().restoreState(from: data).position == ConversationStateTests.prompt.count)
    }
}
