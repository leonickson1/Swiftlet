import Foundation
import NIOHTTP1
import NIOPosix
import Testing
@testable import SwiftletCore
@testable import SwiftletServer

@Suite struct CacheFlushTests {
    private final class ProbeModel: InferenceModel, @unchecked Sendable {
        enum ProbeError: Swift.Error { case cancellationTimedOut }
        let config: QwenConfig
        let modelDir: URL
        private let lock = NSLock()
        private var _calls: [[Int]] = []
        private var blockNext = false
        private var _blocked = false

        init() throws {
            modelDir = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("fixtures/tiny-model")
            config = try QwenConfig(url: modelDir.appendingPathComponent("config.json"))
        }

        var calls: [[Int]] { locked { _calls } }
        var blocked: Bool { locked { _blocked } }
        func blockNextStep() { locked { blockNext = true } }

        func step(_ tokens: [Int], state: QwenCPUModel.DecodeState) throws -> [Float] {
            try step(tokens, state: state, shouldCancel: { false })
        }

        func step(_ tokens: [Int], state: QwenCPUModel.DecodeState,
                  shouldCancel: () -> Bool) throws -> [Float] {
            let shouldBlock = locked {
                _calls.append(tokens)
                let value = blockNext
                blockNext = false
                _blocked = value
                return value
            }
            if shouldBlock {
                defer { locked { _blocked = false } }
                let deadline = Date().addingTimeInterval(3)
                while !shouldCancel() {
                    if Date() >= deadline { throw ProbeError.cancellationTimedOut }
                    Thread.sleep(forTimeInterval: 0.001)
                }
            }
            if shouldCancel() { throw GenerationInterruption.cancelled }
            state.position += tokens.count
            var logits = [Float](repeating: -.infinity, count: config.vocabSize)
            logits[65] = 1
            return logits
        }

        private func locked<T>(_ body: () -> T) -> T {
            lock.lock()
            defer { lock.unlock() }
            return body()
        }
    }

    private func makeSession(_ model: ProbeModel) -> SwiftletSession {
        SwiftletSession(
            testingModel: model, modelDir: model.modelDir,
            encodeText: { _ in [30] },
            decodeTokens: { _ in "A" },
            renderMessages: { messages in
                switch messages.count {
                case 1: return [10]
                case 3: return [10, 65, 20]
                default: return [10, 65, 20, 65, 40]
                }
            }
        )
    }

    /// Exercise the real HTTP pipeline on an ephemeral loopback port; only
    /// model inference and tokenization are replaced with deterministic probes.
    private func withServer(
        session: SwiftletSession, _ body: (URL) async throws -> Void
    ) async throws {
        let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
        let queue = DispatchQueue(label: "swiftlet.test.generation")
        do {
            let server = try await ServerBootstrap(group: group)
                .childChannelInitializer { channel in
                    channel.pipeline.configureHTTPServerPipeline().flatMap {
                        channel.pipeline.addHandler(HTTPHandler(
                            session: session, modelName: "test", generationQueue: queue))
                    }
                }
                .bind(host: "127.0.0.1", port: 0).get()
            do {
                let port = try #require(server.localAddress?.port)
                try await body(URL(string: "http://127.0.0.1:\(port)")!)
            } catch {
                try await server.close().get()
                throw error
            }
            try await server.close().get()
        } catch {
            try await group.shutdownGracefully()
            throw error
        }
        try await group.shutdownGracefully()
    }

    private func request(
        _ base: URL, path: String, method: String = "POST", body: [String: Any]? = nil
    ) -> URLRequest {
        var request = URLRequest(url: base.appendingPathComponent(path))
        request.httpMethod = method
        request.timeoutInterval = 5
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = jsonData(body)
        }
        return request
    }

    private func chat(_ base: URL, messages: [[String: String]], stream: Bool = false) -> URLRequest {
        request(base, path: "v1/chat/completions", body: [
            "messages": messages, "max_tokens": 1, "temperature": 0,
            "min_tokens": 0, "stream": stream,
        ])
    }

    @Test func postFlushDropsWarmConversationAndGetDoesNot() async throws {
        let model = try ProbeModel()
        let session = makeSession(model)
        try await withServer(session: session) { base in
            let first = [["role": "user", "content": "one"]]
            let next = first + [
                ["role": "assistant", "content": "A"], ["role": "user", "content": "next"],
            ]
            let last = next + [
                ["role": "assistant", "content": "A"], ["role": "user", "content": "last"],
            ]
            let (_, firstResponse) = try await URLSession.shared.data(for: chat(base, messages: first))
            #expect((firstResponse as? HTTPURLResponse)?.statusCode == 200)

            let (_, getResponse) = try await URLSession.shared.data(for:
                request(base, path: "v1/cache/flush", method: "GET"))
            #expect((getResponse as? HTTPURLResponse)?.statusCode == 404)
            _ = try await URLSession.shared.data(for: chat(base, messages: next))
            #expect(model.calls == [[10], [65], [30], [65]], "GET must leave the warm continuation intact")

            let (data, response) = try await URLSession.shared.data(for:
                request(base, path: "v1/cache/flush"))
            #expect((response as? HTTPURLResponse)?.statusCode == 200)
            let json = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            #expect(json["status"] as? String == "ok")
            #expect(json["flushed"] as? Bool == true)

            _ = try await URLSession.shared.data(for: chat(base, messages: last))
            #expect(model.calls == [[10], [65], [30], [65], [10, 65, 20, 65, 40], [65]],
                    "POST must discard the warm state and prefill the complete transcript")
        }
    }

    @Test func flushCancelsStreamingGenerationAndNextRequestIsCold() async throws {
        let model = try ProbeModel()
        let session = makeSession(model)
        try await withServer(session: session) { base in
            let first = [["role": "user", "content": "one"]]
            _ = try await URLSession.shared.data(for: chat(base, messages: first))
            let next = first + [
                ["role": "assistant", "content": "A"], ["role": "user", "content": "next"],
            ]
            model.blockNextStep()
            let streamRequest = chat(base, messages: next, stream: true)
            let stream = Task {
                do {
                    _ = try await URLSession.shared.data(for: streamRequest)
                } catch let error as URLError {
                    // An explicit cancellation closes the interrupted SSE
                    // connection instead of inventing an OpenAI finish reason.
                    #expect(error.code == .networkConnectionLost)
                }
            }
            defer { stream.cancel() }
            let deadline = Date().addingTimeInterval(1)
            while !model.blocked, Date() < deadline {
                try await Task<Never, Never>.sleep(nanoseconds: 1_000_000)
            }
            try #require(model.blocked)

            let (_, response) = try await URLSession.shared.data(for:
                request(base, path: "v1/cache/flush"))
            #expect((response as? HTTPURLResponse)?.statusCode == 200)
            try await stream.value
            #expect(session.lastMetrics.finishReason == .cancelled)

            _ = try await URLSession.shared.data(for: chat(base, messages: next))
            #expect(model.calls == [[10], [65], [30], [10, 65, 20], [65]],
                    "the request after cancellation must prefill the complete transcript")
        }
    }
}
