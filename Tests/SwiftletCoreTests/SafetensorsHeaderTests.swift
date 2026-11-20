import Foundation
import Testing
@testable import SwiftletCore

@Suite struct SafetensorsHeaderTests {
    static func fixture(header: String, payload: Data = Data(), headerLength: UInt64? = nil) throws -> URL {
        let json = Data(header.utf8)
        var length = (headerLength ?? UInt64(json.count)).littleEndian
        var file = Data(bytes: &length, count: 8)
        file.append(json)
        file.append(payload)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("header-\(UUID().uuidString).safetensors")
        try file.write(to: url)
        return url
    }

    static func expectMalformed(_ url: URL, naming fields: [String]) {
        do {
            _ = try SafetensorsFile(url: url)
            Issue.record("a malformed safetensors header was opened")
        } catch let error as SafetensorsFile.Error {
            switch error {
            case .malformedHeader:
                let text = String(describing: error)
                for field in fields {
                    #expect(text.contains(field), "the refusal must name \(field): \(text)")
                }
            default:
                Issue.record("refused with \(error), not malformedHeader")
            }
        } catch {
            Issue.record("refused with \(error), not malformedHeader")
        }
    }

    /// Both narrowing UInt64 and adding the eight-byte prefix used to trap.
    @Test(arguments: [UInt64(1) << 63, UInt64.max, UInt64(Int.max), 4096])
    func headerLengthIsRefusedBeforeArithmetic(_ length: UInt64) throws {
        let url = try Self.fixture(header: "{}", headerLength: length)
        defer { try? FileManager.default.removeItem(at: url) }
        Self.expectMalformed(url, naming: ["header length", "\(length)"])
    }

    /// Opening must refuse these ranges before Range or any byte view is made.
    @Test(arguments: [[100, 50], [-1, 3], [0, 5], [0, Int.max]])
    func dataOffsetsAreRefusedByTensorName(_ offsets: [Int]) throws {
        let url = try Self.fixture(
            header: """
            {"weight":{"dtype":"U8","shape":[4],"data_offsets":[\(offsets[0]),\(offsets[1])]}}
            """, payload: Data([1, 2, 3, 4]))
        defer { try? FileManager.default.removeItem(at: url) }
        Self.expectMalformed(url, naming: ["weight", "data_offsets"])
    }

    /// A small file is enough to exercise shape and dtype byte-count overflow.
    @Test(arguments: [
        ("F32", "[2]"), ("U8", "[\(Int.max),2]"), ("F64", "[\(Int.max)]"),
        ("U8", "[-1,-1]"), ("U8", "[0,-1]"), ("unknown", "[4]"),
    ])
    func tensorByteCountMustMatchShape(_ dtype: String, _ shape: String) throws {
        let url = try Self.fixture(
            header: """
            {"weight":{"dtype":"\(dtype)","shape":\(shape),"data_offsets":[0,4]}}
            """, payload: Data([1, 2, 3, 4]))
        defer { try? FileManager.default.removeItem(at: url) }
        Self.expectMalformed(url, naming: ["weight"])
    }

    /// Scalars, zero-sized tensors and ranges ending exactly at EOF stay valid.
    @Test func scalarAndEmptyTensorAtEOFStillOpen() throws {
        let url = try Self.fixture(header: """
            {"scalar":{"dtype":"U8","shape":[],"data_offsets":[0,1]},
             "empty":{"dtype":"F32","shape":[2,0],"data_offsets":[1,1]}}
            """, payload: Data([42]))
        defer { try? FileManager.default.removeItem(at: url) }
        let file = try SafetensorsFile(url: url)
        #expect(try file.raw("scalar").bytes == Data([42]))
        #expect(try file.raw("empty").bytes.isEmpty)
        #expect(try file.withRawBytes("scalar") { _, bytes in Array(bytes) } == [42])
        let empty = try file.info("empty")
        #expect(file.withTensorBytes(empty, byteRange: 0..<0) { $0.isEmpty })
    }
}
