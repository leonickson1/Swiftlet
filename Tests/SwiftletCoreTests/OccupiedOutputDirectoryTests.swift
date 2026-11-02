import Foundation
import Testing
@testable import SwiftletCore

/// An output directory that already holds a checkpoint is refused by name
/// before anything is written into it; an interrupted streaming install is
/// resumed only when the directory still carries its `.install-progress.json`
/// sidecar. Before this both producers began with
/// `createDirectory(withIntermediateDirectories: true)` and wrote straight
/// over whatever was there: `--output` aimed at a Hugging Face snapshot
/// truncated its `model.safetensors` to the dense file, `--output` aimed at
/// a finished container overwrote its manifest, and the streaming installer
/// could not tell an interrupted install of its own from a checkpoint
/// someone else put there, so it "resumed" into either.
@Suite struct OccupiedOutputDirectoryTests {
    static let fixturesDir = MetalModelTests.fixturesDir
    static let source = fixturesDir.appendingPathComponent("tiny-model-q4")
    static let sidecar = Qpack.installProgressSidecar

    static func freshOutput(_ tag: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("occupied-\(tag)-\(UUID().uuidString).qpack")
    }

    /// An output directory pre-populated with `files` (relative paths, each
    /// holding `marker` bytes so an overwrite is detectable).
    static func occupied(_ tag: String, files: [String], marker: Data = Data("occupant".utf8)) throws -> URL {
        let out = freshOutput(tag)
        for file in files {
            let url = out.appendingPathComponent(file)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try marker.write(to: url)
        }
        return out
    }

    static func expectOccupied(by file: String, in dir: URL, _ body: () throws -> Void) {
        do {
            try body()
            Issue.record("an output directory holding \(file) was written into")
        } catch Qpack.Error.occupiedOutput(let namedDir, let namedFile) {
            #expect(namedDir == dir.path, "the refusal must name the directory")
            #expect(namedFile == file, "the refusal must name the offending file")
            let text = Qpack.Error.occupiedOutput(dir: namedDir, file: namedFile).description
            #expect(text.contains(file) && text.contains("already holds a checkpoint"))
        } catch {
            Issue.record("refused with \(error), not as an occupied output directory")
        }
    }

    static func expectInterruptedInstall(in dir: URL, _ body: () throws -> Void) {
        do {
            try body()
            Issue.record("a directory holding an interrupted install was repacked into")
        } catch Qpack.Error.interruptedInstall(let namedDir) {
            #expect(namedDir == dir.path)
            #expect(Qpack.Error.interruptedInstall(namedDir).description.contains(sidecar))
        } catch {
            Issue.record("refused with \(error), not as an interrupted install")
        }
    }

    static func repack(into out: URL) throws {
        var repacker = QpackRepacker(checkpointDir: source, outputDir: out)
        repacker.log = { _ in }
        try repacker.repack()
    }

    static func install(from src: URL = source, into out: URL, cancelOnFirstPoll: Bool = false) throws {
        let installer = StreamingInstaller(source: .localDirectory(src), outputDir: out)
        installer.log = { _ in }
        if cancelOnFirstPoll { installer.shouldCancel = { true } }
        try installer.install()
    }

    static func exists(_ relative: String, in dir: URL) -> Bool {
        FileManager.default.fileExists(atPath: dir.appendingPathComponent(relative).path)
    }

    static func manifest(of dir: URL) throws -> Qpack.Manifest {
        try JSONDecoder().decode(Qpack.Manifest.self, from: Data(contentsOf: dir.appendingPathComponent("manifest.json")))
    }

    // MARK: (a) HF shards present

    /// `--output` aimed at a raw checkpoint: the single-shard and the sharded
    /// spellings are both refused by the shard's name, and the shard keeps
    /// its bytes.
    @Test func repackerRefusesAnOutputDirectoryHoldingHFShards() throws {
        let single = try Self.occupied("shard", files: ["config.json", "model.safetensors"])
        let sharded = try Self.occupied("sharded", files: [
            "model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors",
        ])
        defer {
            try? FileManager.default.removeItem(at: single)
            try? FileManager.default.removeItem(at: sharded)
        }
        Self.expectOccupied(by: "model.safetensors", in: single) { try Self.repack(into: single) }
        #expect(try Data(contentsOf: single.appendingPathComponent("model.safetensors")) == Data("occupant".utf8),
                "the occupant's shard must not be touched")
        #expect(!Self.exists("packed_experts", in: single))
        #expect(!Self.exists("manifest.json", in: single))

        Self.expectOccupied(by: "model-00001-of-00002.safetensors", in: sharded) { try Self.repack(into: sharded) }
        #expect(!Self.exists("packed_experts", in: sharded))
    }

    /// The streaming installer refuses the same directory before it writes
    /// `config.json`, a layer file, or the dense file.
    @Test func installerRefusesAnOutputDirectoryHoldingHFShards() throws {
        let out = try Self.occupied("shard-stream", files: ["model.safetensors"])
        defer { try? FileManager.default.removeItem(at: out) }
        Self.expectOccupied(by: "model.safetensors", in: out) { try Self.install(into: out) }
        for file in ["config.json", "manifest.json", "packed_experts", Self.sidecar] {
            #expect(!Self.exists(file, in: out), "\(file) was written into a refused directory")
        }
        #expect(try Data(contentsOf: out.appendingPathComponent("model.safetensors")) == Data("occupant".utf8))
    }

    // MARK: (b) index or manifest present

    /// A finished container (`manifest.json`), a sharded checkpoint's index,
    /// and a container's expert layout or blob are each refused by name; the
    /// index and the manifest are named ahead of the shards beside them.
    @Test func indexOrManifestIsNamedBeforeTheShards() throws {
        let cases: [(files: [String], named: String)] = [
            (["manifest.json"], "manifest.json"),
            (["manifest.json", "model.safetensors", "packed_experts/layer_00.bin"], "manifest.json"),
            (["model.safetensors.index.json", "model-00001-of-00002.safetensors"], "model.safetensors.index.json"),
            (["packed_experts/layout.json"], "packed_experts/layout.json"),
            (["packed_experts/layer_00.bin"], "packed_experts/layer_00.bin"),
        ]
        for (files, named) in cases {
            let forRepack = try Self.occupied("marker", files: files)
            let forInstall = try Self.occupied("marker-stream", files: files)
            defer {
                try? FileManager.default.removeItem(at: forRepack)
                try? FileManager.default.removeItem(at: forInstall)
            }
            Self.expectOccupied(by: named, in: forRepack) { try Self.repack(into: forRepack) }
            Self.expectOccupied(by: named, in: forInstall) { try Self.install(into: forInstall) }
            for file in files {
                #expect(try Data(contentsOf: forRepack.appendingPathComponent(file)) == Data("occupant".utf8),
                        "\(file) was overwritten by the repacker")
                #expect(try Data(contentsOf: forInstall.appendingPathComponent(file)) == Data("occupant".utf8),
                        "\(file) was overwritten by the installer")
            }
        }
    }

    /// A finished container is finished: the manifest is refused even when a
    /// stale sidecar sits beside it.
    @Test func aCompleteContainerIsRefusedEvenBesideAStaleSidecar() throws {
        let out = try Self.occupied("stale", files: ["manifest.json", Self.sidecar])
        defer { try? FileManager.default.removeItem(at: out) }
        Self.expectOccupied(by: "manifest.json", in: out) { try Self.install(into: out) }
        Self.expectOccupied(by: "manifest.json", in: out) { try Self.repack(into: out) }
    }

    // MARK: (c) sidecar only

    /// Only the sidecar: the streaming installer resumes (here, from nothing)
    /// and finishes the container; the sidecar is gone when it is done.
    @Test func installerResumesWhenOnlyTheSidecarIsPresent() throws {
        let out = try Self.occupied("sidecar-only", files: [Self.sidecar], marker: Data("{}".utf8))
        let clean = Self.freshOutput("clean")
        defer {
            try? FileManager.default.removeItem(at: out)
            try? FileManager.default.removeItem(at: clean)
        }
        try Self.install(into: out)
        try Self.install(into: clean)
        #expect(Self.exists("manifest.json", in: out))
        #expect(!Self.exists(Self.sidecar, in: out), "a finished install must not leave its sidecar behind")
        #expect(try Self.manifest(of: out).files == Self.manifest(of: clean).files)
        _ = try Qpack.verify(containerDir: out)
    }

    /// The repacker has no resume: a directory holding an interrupted
    /// streaming install is refused, with the sidecar named.
    @Test func repackerRefusesAnInterruptedInstall() throws {
        let out = try Self.occupied("sidecar-repack", files: [Self.sidecar], marker: Data("{}".utf8))
        defer { try? FileManager.default.removeItem(at: out) }
        Self.expectInterruptedInstall(in: out) { try Self.repack(into: out) }
        #expect(!Self.exists("packed_experts", in: out))
    }

    // MARK: an interrupted install of its own

    /// The property the sidecar protects: an install interrupted after the
    /// first bytes landed still resumes and finishes as a verifiable
    /// container, byte-for-byte the manifest a clean install produces.
    @Test func anInterruptedStreamingInstallStillResumes() throws {
        let out = Self.freshOutput("interrupted")
        let clean = Self.freshOutput("clean")
        defer {
            try? FileManager.default.removeItem(at: out)
            try? FileManager.default.removeItem(at: clean)
        }
        do {
            try Self.install(into: out, cancelOnFirstPoll: true)
            Issue.record("the cancelled install ran to completion")
        } catch StreamingInstaller.Error.cancelled {
        }
        #expect(Self.exists(Self.sidecar, in: out), "an interrupted install must leave its sidecar")
        #expect(!Self.exists("manifest.json", in: out))
        #expect(Self.exists("model.safetensors", in: out), "the dense file was already opened when the cancel landed")

        try Self.install(into: out)
        try Self.install(into: clean)
        #expect(!Self.exists(Self.sidecar, in: out))
        #expect(try Self.manifest(of: out).files == Self.manifest(of: clean).files)
        _ = try Qpack.verify(containerDir: out)
    }

    /// The direct-download path (a source that is already a container) has
    /// the same contract: an interrupted download resumes through the
    /// sidecar, and a directory holding shards without one is refused.
    @Test func anInterruptedDirectDownloadStillResumes() throws {
        let container = Self.freshOutput("container")
        let out = Self.freshOutput("direct")
        let occupied = try Self.occupied("direct-occupied", files: ["model.safetensors"])
        defer {
            try? FileManager.default.removeItem(at: container)
            try? FileManager.default.removeItem(at: out)
            try? FileManager.default.removeItem(at: occupied)
        }
        try Self.repack(into: container)

        do {
            try Self.install(from: container, into: out, cancelOnFirstPoll: true)
            Issue.record("the cancelled download ran to completion")
        } catch StreamingInstaller.Error.cancelled {
        }
        #expect(Self.exists(Self.sidecar, in: out), "an interrupted direct download must leave its sidecar")
        #expect(!Self.exists("manifest.json", in: out))

        try Self.install(from: container, into: out)
        #expect(!Self.exists(Self.sidecar, in: out))
        #expect(try Self.manifest(of: out).files == Self.manifest(of: container).files)
        _ = try Qpack.verify(containerDir: out)

        Self.expectOccupied(by: "model.safetensors", in: occupied) {
            try Self.install(from: container, into: occupied)
        }
    }

    // MARK: still accepted

    /// An existing empty directory, and one holding only aux files, are not
    /// checkpoints and are written into as before.
    @Test func anEmptyOrAuxOnlyDirectoryIsAccepted() throws {
        let empty = Self.freshOutput("empty")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        let auxOnly = try Self.occupied("aux", files: ["config.json", "tokenizer.json"])
        defer {
            try? FileManager.default.removeItem(at: empty)
            try? FileManager.default.removeItem(at: auxOnly)
        }
        try Self.repack(into: empty)
        #expect(Self.exists("manifest.json", in: empty))
        try Self.install(into: auxOnly)
        #expect(Self.exists("manifest.json", in: auxOnly))
        _ = try Qpack.verify(containerDir: auxOnly)
    }
}
