import Foundation
import Testing
@testable import SwiftletCore

/// Serve real header bytes to the installer's HTTP path without a live server.
private final class SafetensorsHeaderProtocol: URLProtocol {
    static let lock = NSLock()
    nonisolated(unsafe) static var files: [String: [String: Data]] = [:]

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.host?.hasSuffix(".safetensors.invalid") == true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let url = request.url!
        Self.lock.lock()
        let data = Self.files[url.host!]?[url.lastPathComponent]
        Self.lock.unlock()
        var body = data ?? Data()
        var status = data == nil ? 404 : 200
        var headers: [String: String] = [:]
        if let range = request.value(forHTTPHeaderField: "Range"), let data {
            let parts = range.dropFirst("bytes=".count).split(separator: "-")
            let lo = Int(parts[0])!, hi = Int(parts[1])!
            if lo >= 0, hi >= lo, hi < data.count {
                status = 206
                body = data.subdata(in: lo..<(hi + 1))
                headers["Content-Range"] = "bytes \(lo)-\(hi)/\(data.count)"
            } else {
                status = 416
                body = Data()
            }
        }
        headers["Content-Length"] = "\(body.count)"
        let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

@Suite(.serialized) struct StreamingSafetensorsHeaderTests {
    static func withInstaller(header: String, length: UInt64? = nil,
                              _ body: (StreamingInstaller, URL) throws -> Void) throws {
        let url = try SafetensorsHeaderTests.fixture(
            header: header, payload: Data([1, 2, 3, 4]), headerLength: length)
        defer { try? FileManager.default.removeItem(at: url) }
        let config = try Data(contentsOf: MetalModelTests.fixturesDir
            .appendingPathComponent("tiny-model-q4/config.json"))
        let host = "\(UUID().uuidString.lowercased()).safetensors.invalid"
        let bytes = try Data(contentsOf: url)
        SafetensorsHeaderProtocol.lock.lock()
        SafetensorsHeaderProtocol.files[host] = ["config.json": config, "model.safetensors": bytes]
        SafetensorsHeaderProtocol.lock.unlock()
        #expect(URLProtocol.registerClass(SafetensorsHeaderProtocol.self))
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("remote-header-\(UUID().uuidString).qpack")
        defer {
            SafetensorsHeaderProtocol.lock.lock()
            SafetensorsHeaderProtocol.files.removeValue(forKey: host)
            SafetensorsHeaderProtocol.lock.unlock()
            URLProtocol.unregisterClass(SafetensorsHeaderProtocol.self)
            try? FileManager.default.removeItem(at: out)
        }
        let installer = StreamingInstaller(source: .baseURL("https://\(host)"), outputDir: out)
        installer.log = { _ in }
        try body(installer, out)
    }

    static func expectBadHeader(_ installer: StreamingInstaller, out: URL, naming fields: [String]) {
        do {
            try installer.install()
            Issue.record("a malformed remote header was installed")
        } catch StreamingInstaller.Error.badHeader(let why) {
            for field in ["model.safetensors"] + fields {
                #expect(why.contains(field), "the refusal must name \(field): \(why)")
            }
        } catch {
            Issue.record("refused with \(error), not badHeader")
        }
        #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent("manifest.json").path))
        #expect(!FileManager.default.fileExists(atPath: out.appendingPathComponent("model.safetensors").path))
    }

    @Test(arguments: [UInt64(1) << 63, UInt64(Int.max), 4096])
    func headerLengthIsRefusedBeforeRequest(_ length: UInt64) throws {
        try Self.withInstaller(header: "{}", length: length) { installer, out in
            Self.expectBadHeader(installer, out: out, naming: ["header length", "\(length)"])
        }
    }

    @Test func tensorPastEOFIsRefused() throws {
        try Self.withInstaller(header: """
            {"weight":{"dtype":"U8","shape":[5],"data_offsets":[0,5]}}
            """) { installer, out in
            Self.expectBadHeader(installer, out: out, naming: ["weight", "data_offsets"])
        }
    }

    @Test(arguments: [
        "{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[100,50]}",
        "{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[-1,3]}",
        "{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}",
        "{\"dtype\":\"F64\",\"shape\":[9223372036854775807],\"data_offsets\":[0,4]}",
    ])
    func malformedTensorIsRefusedByName(_ entry: String) throws {
        try Self.withInstaller(header: "{\"weight\":\(entry)}") { installer, out in
            Self.expectBadHeader(installer, out: out, naming: ["weight"])
        }
    }

    /// The header itself is valid; planning then refuses the lack of experts.
    @Test func validTensorEndingAtEOFReachesPlanning() throws {
        try Self.withInstaller(header: """
            {"weight":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}
            """) { installer, _ in
            do {
                try installer.install()
                Issue.record("a header without stacked experts was installed")
            } catch StreamingInstaller.Error.notMLXCheckpoint {
                // Valid header reached the existing expert-layout check.
            } catch {
                Issue.record("valid header refused before planning: \(error)")
            }
        }
    }
}
