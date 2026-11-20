import CryptoKit
import Foundation
import Network
import Testing
@testable import SwiftletCore

/// Exercise URLSession's real response callbacks, not just a header parser:
/// a refused response must never reach the installer's pwrite handler.
@Suite struct StreamingInstallerRangeTests {
    static let manifest = Data(#"{"magic":"QPACK","version":1,"modelName":"tiny","sourceCheckpoint":"test","files":{"model.safetensors":6}}"#.utf8)

    static func output(partial: Data = Data("abc".utf8)) throws -> URL {
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("range-install-\(UUID().uuidString).qpack")
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: out.appendingPathComponent(Qpack.installProgressSidecar))
        try partial.write(to: out.appendingPathComponent("model.safetensors"))
        return out
    }

    /// A size-only install used to accept these responses as a complete file.
    /// Check the partial bytes and the named refusal, including that a bad
    /// range is terminal rather than a transient network hiccup.
    @Test func resumeRefusesInvalidContentRangeBeforeWriting() throws {
        let cases: [(status: Int, header: String?)] = [
            (206, "bytes 0-2/6"),
            (206, nil),
            (206, "invalid"),
            (206, "bytes 3-5/7"),
            (206, "bytes 3-6/6"),
            (206, "bytes 3-2/6"),
            (206, "bytes 3-5/*"),
            (206, "items 3-5/6"),
            (206, "bytes +3-5/6"),
            (206, "bytes 3-5/6 trailing"),
            (206, "bytes 3-5/999999999999999999999999999999"),
            (200, nil),
        ]
        for (status, header) in cases {
            let server = try RangeHTTPServer { request in
                switch request.path {
                case "/manifest.json": return .init(body: Self.manifest)
                case "/model.safetensors":
                    #expect(request.range == "bytes=3-")
                    return .init(status: status, contentRange: header, body: Data("XYZ".utf8))
                default: return .init(status: 404)
                }
            }
            defer { server.stop() }
            let out = try Self.output()
            defer { try? FileManager.default.removeItem(at: out) }
            let installer = StreamingInstaller(source: .baseURL(server.baseURL), outputDir: out)
            var hiccups = 0
            installer.log = { if $0.contains("network hiccup") { hiccups += 1 } }
            do {
                try installer.install()
                Issue.record("resume accepted status \(status), Content-Range \(header ?? "absent")")
            } catch {
                #expect(error is StreamingInstaller.Error)
                let text = error.localizedDescription
                if status == 206 {
                    #expect(text.contains("Content-Range"))
                    #expect(text.contains("bytes 3-*/6"), "name the expected cursor and total")
                    #expect(text.contains(header ?? "absent"), "name the received header")
                } else {
                    #expect(text.contains("Range") && text.contains("200"))
                    #expect(text.contains("bytes=3-"))
                }
            }
            #expect(hiccups == 0, "a protocol refusal must not enter the retry loop")
            #expect(try Data(contentsOf: out.appendingPathComponent("model.safetensors")) == Data("abc".utf8),
                    "no response bytes may be appended for \(header ?? "absent")")
            #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent("manifest.json").path))
            #expect(FileManager.default.fileExists(atPath: out.appendingPathComponent(Qpack.installProgressSidecar).path))
        }
    }

    /// The accepted range still completes, and SHA-256 still checks the
    /// whole resumed file (including the prefix that was already on disk).
    @Test func matching206ResumeStillVerifiesTheWholeFile() throws {
        let hash = "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721"
        for prefix in ["abc", "bad"] {
            let server = try RangeHTTPServer { request in
                switch request.path {
                case "/manifest.json": return .init(body: Self.manifest)
                case "/hashes.json":
                    return .init(body: Data("{\"files\":{\"model.safetensors\":\"\(hash)\"}}".utf8))
                case "/model.safetensors":
                    #expect(request.range == "bytes=3-")
                    return .init(status: 206, contentRange: "bytes 3-5/6", body: Data("def".utf8))
                default: return .init(status: 404)
                }
            }
            defer { server.stop() }
            let out = try Self.output(partial: Data(prefix.utf8))
            defer { try? FileManager.default.removeItem(at: out) }
            let installer = StreamingInstaller(source: .baseURL(server.baseURL), outputDir: out)
            installer.log = { _ in }
            if prefix == "abc" {
                try installer.install()
                let downloaded = try Data(contentsOf: out.appendingPathComponent("model.safetensors"))
                #expect(downloaded == Data("abcdef".utf8))
                #expect(SHA256.hash(data: downloaded).map { String(format: "%02x", $0) }.joined() == hash)
                #expect(try Data(contentsOf: out.appendingPathComponent("manifest.json")) == Self.manifest)
                #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent(Qpack.installProgressSidecar).path))
            } else {
                do {
                    try installer.install()
                    Issue.record("a resumed file with a corrupt prefix passed SHA-256")
                } catch StreamingInstaller.Error.hashMismatch(let name) {
                    #expect(name == "model.safetensors")
                }
                #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent("model.safetensors").path))
                #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent("manifest.json").path))
            }
        }
    }
}

/// One HTTP response per connection, bound to loopback on an ephemeral port.
/// No process-wide URLProtocol registration or external server is needed.
private final class RangeHTTPServer: @unchecked Sendable {
    struct Request {
        let path: String
        let range: String?
    }

    struct Response {
        var status = 200
        var contentRange: String? = nil
        var body = Data()
    }

    private let listener: NWListener
    private let queue = DispatchQueue(label: "swiftlet.range-test-http")
    private let respond: @Sendable (Request) -> Response
    var baseURL: String { "http://127.0.0.1:\(listener.port!.rawValue)" }

    init(respond: @escaping @Sendable (Request) -> Response) throws {
        self.respond = respond
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1", port: .any)
        let listener = try NWListener(using: parameters)
        self.listener = listener
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { connection.cancel(); return }
            connection.start(queue: self.queue)
            self.receive(connection, buffered: Data())
        }
        let ready = DispatchSemaphore(value: 0)
        listener.stateUpdateHandler = { state in
            switch state {
            case .ready, .failed: ready.signal()
            default: break
            }
        }
        listener.start(queue: queue)
        guard ready.wait(timeout: .now() + 5) == .success, listener.port != nil else {
            listener.cancel()
            throw NSError(domain: "RangeHTTPServer", code: 1)
        }
    }

    func stop() { listener.cancel() }

    private func receive(_ connection: NWConnection, buffered: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 16_384) { [self] data, _, complete, error in
            var bytes = buffered
            if let data { bytes.append(data) }
            guard let end = bytes.range(of: Data("\r\n\r\n".utf8)) else {
                if complete || error != nil { connection.cancel() }
                else { self.receive(connection, buffered: bytes) }
                return
            }
            let lines = String(decoding: bytes[..<end.lowerBound], as: UTF8.self).components(separatedBy: "\r\n")
            let path = lines[0].split(separator: " ").dropFirst().first.map(String.init) ?? ""
            let range = lines.dropFirst().first { $0.lowercased().hasPrefix("range:") }
                .map { String($0.dropFirst(6)).trimmingCharacters(in: .whitespaces) }
            let response = respond(Request(path: path, range: range))
            var header = "HTTP/1.1 \(response.status) Test\r\nContent-Length: \(response.body.count)\r\nConnection: close\r\n"
            if let contentRange = response.contentRange { header += "Content-Range: \(contentRange)\r\n" }
            var reply = Data((header + "\r\n").utf8)
            reply.append(response.body)
            connection.send(content: reply, contentContext: .finalMessage, isComplete: true,
                            completion: .contentProcessed { _ in connection.cancel() })
        }
    }
}
