import Foundation
import Testing
import SwiftletCore
@testable import SwiftletServer

/// The built `swiftlet-server` against models that cannot open: one stderr
/// line naming the kind, and the kind's exit code. Neither case reaches
/// Metal (the tokenizer loader and `QwenConfig` refuse first), so this runs
/// on any Mac that can build the package.
@Suite struct ServerStartupTests {
    private final class BundleMarker {}

    private static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    /// `swift build --build-tests` places every product beside the test
    /// bundle, so the server binary is one directory above it.
    private static var serverBinary: URL {
        Bundle(for: BundleMarker.self).bundleURL
            .deletingLastPathComponent()
            .appendingPathComponent("swiftlet-server")
    }

    private struct Run {
        let status: Int32
        let stderr: String
        let stdout: String
    }

    private final class Captured: @unchecked Sendable {
        private let lock = NSLock()
        private var _stdout = ""
        private var _stderr = ""
        var stdout: String { lock.lock(); defer { lock.unlock() }; return _stdout }
        var stderr: String { lock.lock(); defer { lock.unlock() }; return _stderr }
        func set(_ value: String, stderr: Bool) {
            lock.lock()
            if stderr { _stderr = value } else { _stdout = value }
            lock.unlock()
        }
    }

    private func run(_ arguments: [String], timeout: TimeInterval = 60) throws -> Run {
        let binary = Self.serverBinary
        try #require(
            FileManager.default.isExecutableFile(atPath: binary.path),
            "swiftlet-server is not built at \(binary.path)"
        )
        let process = Process()
        process.executableURL = binary
        process.arguments = arguments
        let out = Pipe(), err = Pipe()
        process.standardOutput = out
        process.standardError = err
        // Drain both pipes off-thread: reading to EOF inline would wait on a
        // server that (wrongly) came up and is listening, instead of on the
        // deadline below.
        let drained = DispatchGroup()
        let captured = Captured()
        for (handle, isStderr) in [(out.fileHandleForReading, false), (err.fileHandleForReading, true)] {
            drained.enter()
            DispatchQueue.global().async {
                let data = handle.readDataToEndOfFile()
                captured.set(String(decoding: data, as: UTF8.self), stderr: isStderr)
                drained.leave()
            }
        }
        try process.run()
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if process.isRunning {
            process.terminate()
            Issue.record("swiftlet-server \(arguments) still running after \(timeout)s")
        }
        process.waitUntilExit()
        drained.wait()
        return Run(status: process.terminationStatus, stderr: captured.stderr, stdout: captured.stdout)
    }

    @Test func noModelFlagIsAUsageError() throws {
        let run = try run([])
        #expect(run.status == 2)
        #expect(run.stdout.contains("usage: swiftlet-server --model <dir>"))
        for kind in [StartupFailure.container(""), .config(""), .resource(""), .backend("")] {
            #expect(run.stdout.contains("\(kind.exitCode) \(kind.kindName)"))
        }
    }

    @Test func missingModelDirectoryExitsAsContainer() throws {
        let missing = FileManager.default.temporaryDirectory
            .appendingPathComponent("swiftlet-server-missing-\(UUID().uuidString)")
        let run = try run(["--model", missing.path, "--port", "0"])
        #expect(run.status == StartupFailure.container("").exitCode, Comment(rawValue: run.stderr))
        #expect(run.stderr.contains("swiftlet-server: startup failed (container): "), Comment(rawValue: run.stderr))
        #expect(run.stderr.contains("tokenizer.json"), Comment(rawValue: run.stderr))
    }

    @Test func refusedModelTypeExitsAsConfig() throws {
        let copy = FileManager.default.temporaryDirectory
            .appendingPathComponent("swiftlet-server-model-type-\(UUID().uuidString)")
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

        let run = try run(["--model", copy.path, "--port", "0"])
        #expect(run.status == StartupFailure.config("").exitCode, Comment(rawValue: run.stderr))
        #expect(run.stderr.contains("swiftlet-server: startup failed (config): "), Comment(rawValue: run.stderr))
        #expect(run.stderr.contains("glm5_next"), Comment(rawValue: run.stderr))
    }
}
