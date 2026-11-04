import Foundation
import Testing
import Tokenizers
@testable import SwiftletServer

@Suite struct DeveloperRoleTests {
    private final class BundleMarker {}

    private static let modelDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures/tiny-model")

    /// Exercise the server's mapping with the real tokenizer: a developer
    /// instruction must produce exactly the system prompt's token IDs.
    @Test func developerRendersTheSameTokensAsSystem() async throws {
        let tokenizer = try await AutoTokenizer.from(modelFolder: Self.modelDir)
        let request = try JSONDecoder().decode(ChatRequest.self, from: Data(#"{"messages":[{"role":"developer","content":[{"type":"text","text":"Be "},{"type":"text","text":"brief."}]},{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"tool","content":"ok"},{"role":"user","content":"next"}]}"#.utf8))
        let expected = [
            ["role": "system", "content": "Be brief."],
            ["role": "user", "content": "hi"],
            ["role": "assistant", "content": "hello"],
            ["role": "tool", "content": "ok"],
            ["role": "user", "content": "next"],
        ]
        let messages = chatMessages(request)
        #expect(messages == expected)
        let actualIDs = try tokenizer.applyChatTemplate(messages: messages)
        let expectedIDs = try tokenizer.applyChatTemplate(messages: expected)
        #expect(!expectedIDs.isEmpty)
        #expect(actualIDs == expectedIDs)
    }

    /// The HTTP route must admit developer instructions and keep the warm
    /// prefix even when the client alternates system/developer spellings.
    @Test func developerRequestReturns200AndFollowUpsReuseTheConversation() async throws {
        let binary = Bundle(for: BundleMarker.self).bundleURL
            .deletingLastPathComponent().appendingPathComponent("swiftlet-server")
        try #require(FileManager.default.isExecutableFile(atPath: binary.path))
        let log = FileManager.default.temporaryDirectory
            .appendingPathComponent("swiftlet-developer-\(UUID().uuidString).log")
        try Data().write(to: log)
        defer { try? FileManager.default.removeItem(at: log) }
        let output = try FileHandle(forWritingTo: log)
        defer { try? output.close() }
        let process = Process()
        process.executableURL = binary
        let port = Int.random(in: 49152...65535)
        process.arguments = ["--model", Self.modelDir.path, "--port", "\(port)"]
        process.standardOutput = output
        process.standardError = output
        try process.run()
        defer {
            if process.isRunning { process.terminate() }
            process.waitUntilExit()
        }
        let client = URLSession(configuration: .ephemeral)
        defer { client.invalidateAndCancel() }
        let base = "http://127.0.0.1:\(port)"
        let deadline = Date().addingTimeInterval(30)
        var ready = false
        while process.isRunning, Date() < deadline {
            if let (_, response) = try? await client.data(from: URL(string: base + "/v1/models")!),
               (response as? HTTPURLResponse)?.statusCode == 200 {
                ready = true
                break
            }
            try await Task<Never, Never>.sleep(nanoseconds: 50_000_000)
        }
        try #require(ready, Comment(rawValue: (try? String(contentsOf: log, encoding: .utf8)) ?? "server did not become ready"))

        var history = [
            ["role": "developer", "content": "Be brief."],
            ["role": "user", "content": "hi"],
        ]
        // Start with developer, then echo the same instructions as system,
        // then developer again. Only each new user suffix should be prefilled.
        for turn in 0..<3 {
            if turn > 0 {
                history[0]["role"] = turn == 1 ? "system" : "developer"
                history.append(["role": "user", "content": "ok"])
            }
            var request = URLRequest(url: URL(string: base + "/v1/chat/completions")!)
            request.httpMethod = "POST"
            request.timeoutInterval = 30
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: [
                "messages": history, "max_tokens": 1, "min_tokens": 0, "temperature": 0,
            ])
            let (data, response) = try await client.data(for: request)
            try #require((response as? HTTPURLResponse)?.statusCode == 200,
                         Comment(rawValue: String(decoding: data, as: UTF8.self)))
            let payload = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let usage = try #require(payload["usage"] as? [String: Int])
            #expect(usage["completion_tokens"] == 1)
            if turn > 0 {
                // <|im_end|>\n<|im_start|>user\nok<|im_end|>\n<|im_start|>assistant\n
                // is 23 tokens with tiny-model, regardless of history length.
                #expect(usage["prompt_tokens"] == 23)
            }
            let choices = try #require(payload["choices"] as? [[String: Any]])
            let choice = try #require(choices.first)
            let message = try #require(choice["message"] as? [String: String])
            history.append(["role": "assistant", "content": try #require(message["content"])])
        }
    }
}
