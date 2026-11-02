import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import SwiftletCore

// Loopback OpenAI-compatible Chat Completions server (modeled on
// TurboFieldfare's). One warm model, requests serialized. No auth/TLS: keep it
// on 127.0.0.1.
//
//   swiftlet-server --model <dir> [--port 8080] [--cache-gb 2]

/// Message content per the OpenAI Chat Completions spec: a plain string, an
/// array of parts ([{type:"text",text:"..."}]), or null on tool-call turns.
/// Text parts are joined; null decodes as "". A part of any other type
/// (`image_url`, `input_audio`, ...) is refused by name rather than dropped:
/// this server has no image or audio path, and answering the text left over
/// after a silently dropped image is a reply to a question the client did
/// not ask. The refusal is whole-request, so text beside an image does not
/// rescue it.
struct ChatContent: Decodable {
    struct Part: Decodable { let type: String?; let text: String? }

    /// A content part this server cannot consume, carrying its `type` so the
    /// 400 body can name it. Thrown from decoding; `handleChat` reports it as
    /// a bad request with this description instead of "malformed request".
    struct UnsupportedContentPart: Swift.Error, CustomStringConvertible {
        let type: String
        var description: String {
            "unsupported content part type \"\(type)\": this server accepts text parts only"
        }
    }

    let text: String
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if try c.decodeNil() {
            text = ""
        } else if let s = try? c.decode(String.self) {
            text = s
        } else {
            var joined = ""
            for part in try c.decode([Part].self) {
                if let type = part.type, type != "text" {
                    throw UnsupportedContentPart(type: type)
                }
                // A part that names no type is text when it carries text; a
                // `text` part without `text` is a malformed body, not a
                // foreign part type.
                guard let text = part.text else {
                    throw DecodingError.dataCorruptedError(
                        in: c, debugDescription: "content part has no text")
                }
                joined += text
            }
            text = joined
        }
    }
}

/// OpenAI accepts `stop` as either one string or an array of strings.
struct StopSequences: Decodable {
    let values: [String]

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            values = [value]
        } else {
            values = try container.decode([String].self)
        }
    }
}

/// OpenAI-compatible Chat Completions request body.
struct ChatRequest: Decodable {
    struct Message: Decodable { let role: String; let content: ChatContent }
    let messages: [Message]
    let stream: Bool?
    let max_tokens: Int?
    let max_completion_tokens: Int?
    let temperature: Float?
    let top_p: Float?
    let stop: StopSequences?
    // Repetition controls. `presence_penalty` and `frequency_penalty` are
    // standard OpenAI request fields; `no_repeat_ngram` and `min_tokens` expose
    // the two remaining knobs that have no OpenAI equivalent.
    //
    // These exist for structured output. The defaults are tuned for prose on a
    // quantized checkpoint (issue #13), and JSON is the case they penalise: its
    // structural trigrams (`",\n  "`, `": "`) recur on every key, so
    // noRepeatNGram bans the opening quote of the second key outright, and
    // minNew forces generation past a complete short object. Callers that want
    // machine-readable output need to turn them down; callers that want prose
    // keep today's behaviour by not sending them.
    let presence_penalty: Float?
    let frequency_penalty: Float?
    let no_repeat_ngram: Int?
    let min_tokens: Int?
}

let cliArgs = CommandLine.arguments
func flag(_ name: String) -> String? {
    guard let i = cliArgs.firstIndex(of: name), i + 1 < cliArgs.count else { return nil }
    return cliArgs[i + 1]
}
guard let modelPath = flag("--model") else {
    print("usage: swiftlet-server --model <dir> [--port 8080] [--cache-gb 2]")
    exit(2)
}
let port = Int(flag("--port") ?? "8080") ?? 8080
let cacheGB = Double(flag("--cache-gb") ?? "2") ?? 2
let modelURL = URL(fileURLWithPath: modelPath)

FileHandle.standardError.write(Data("loading model + tokenizer...\n".utf8))
// The same Metal streaming engine as `swiftlet chat`: routed experts stream
// from the .qpack blobs, so a 35B/80B container serves in a few GB of RAM.
// (The CPU model reader cannot serve containers: their experts are not in
// model.safetensors.)
let session = try await SwiftletSession(modelDir: modelURL, cacheBudgetGB: cacheGB)
let modelName: String = {
    let configURL = modelURL.appendingPathComponent("config.json")
    if let cfg = try? JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any],
       let t = cfg["model_type"] as? String { return t }
    return "swiftlet"
}()
// One request at a time: the session mutates shared conversation state.
let generationQueue = DispatchQueue(label: "swiftlet.generation")

/// Owns every generation associated with one HTTP/1 connection. NIO invokes
/// `channelInactive` as soon as a peer disconnects, even while its request is
/// still queued behind another generation, so cancellation reaches both
/// queued and active work instead of waiting for another attempted write.
final class ConnectionGenerationOwner: @unchecked Sendable {
    private struct Job {
        let cancellation: GenerationCancellation
        var heartbeat: RepeatedTask?
    }

    private let lock = NSLock()
    private var jobs: [String: Job] = [:]

    func register(
        id: String,
        cancellation: GenerationCancellation,
        heartbeat: RepeatedTask? = nil
    ) {
        lock.lock()
        jobs[id] = Job(cancellation: cancellation, heartbeat: heartbeat)
        lock.unlock()
    }

    /// Attaches a heartbeat after the initial SSE head is submitted. If that
    /// head has already failed and removed the job, cancel the newly-created
    /// task instead of orphaning it.
    func attachHeartbeat(id: String, heartbeat: RepeatedTask) {
        lock.lock()
        if var job = jobs[id] {
            job.heartbeat = heartbeat
            jobs[id] = job
            lock.unlock()
        } else {
            lock.unlock()
            heartbeat.cancel()
        }
    }

    func stopHeartbeat(id: String) {
        lock.lock()
        let heartbeat = jobs[id]?.heartbeat
        if var job = jobs[id] {
            job.heartbeat = nil
            jobs[id] = job
        }
        lock.unlock()
        heartbeat?.cancel()
    }

    func finish(id: String) {
        lock.lock()
        let job = jobs.removeValue(forKey: id)
        lock.unlock()
        job?.heartbeat?.cancel()
    }

    /// Cancels and removes only the request whose write failed.
    @discardableResult
    func failWrite(id: String) -> Bool {
        lock.lock()
        let job = jobs.removeValue(forKey: id)
        lock.unlock()
        job?.cancellation.cancel()
        job?.heartbeat?.cancel()
        return job != nil
    }

    func cancelAll() {
        lock.lock()
        let pending = Array(jobs.values)
        jobs.removeAll()
        lock.unlock()
        for job in pending {
            job.cancellation.cancel()
            job.heartbeat?.cancel()
        }
    }
}

@Sendable func jsonData(_ obj: [String: Any]) -> Data {
    (try? JSONSerialization.data(withJSONObject: obj)) ?? Data("{}".utf8)
}

/// What this server will not read or run, decided from the request alone
/// before the model is touched. Kept in a type rather than top-level `let`s:
/// `main.swift` globals are not initialised when the module is imported by
/// a test target, a `static let` is.
enum RequestLimits {
    /// Largest request body this server reads, in bytes. A chat request is
    /// a few hundred KB at the most; the cap exists so a client cannot make
    /// the server buffer an arbitrary body before `JSONDecoder` sees it.
    /// The same figure strata chose for its serve path.
    static let maxBodyBytes = 256 << 20
    /// Generation budget when a request names neither `max_tokens` nor
    /// `max_completion_tokens`.
    static let defaultMaxNew = 512
}

/// A request refused at the server, before the model is touched, with the
/// status to send and a message that names the field or header at fault.
struct RequestRefusal: Swift.Error, CustomStringConvertible, Equatable {
    let status: HTTPResponseStatus
    let description: String
}

/// The generation budget a chat request asks for. `max_tokens` wins over
/// `max_completion_tokens` as before; zero or a negative value in either is
/// refused by the field's name. Before this, `max_tokens: 0` ran the prompt
/// through prefill and returned an empty completion with `finish_reason:
/// "length"`, and a negative value reached `ContextWindow`, whose refusal
/// named `requestedMaxNew` (an internal parameter) after the prompt had been
/// rendered and, on a streaming request, after the 200 head was on the wire.
@Sendable func requestedMaxNew(_ request: ChatRequest) throws -> Int {
    for (field, value) in [("max_tokens", request.max_tokens),
                           ("max_completion_tokens", request.max_completion_tokens)] {
        if let value, value <= 0 {
            throw RequestRefusal(
                status: .badRequest,
                description: "\(field) must be a positive integer, got \(value)")
        }
    }
    return request.max_tokens ?? request.max_completion_tokens ?? RequestLimits.defaultMaxNew
}

/// Decided from the request head alone, before any body byte is read: a
/// `Content-Length` over the limit is 413, one that is not a non-negative
/// integer is 400. (NIO's `HTTPServerProtocolErrorHandler`, installed by
/// `configureHTTPServerPipeline()`, already answers an unparsable
/// `Content-Length` with a bare 400 and closes before the head reaches this
/// handler; the rule is kept here so the limit and its spellings live in one
/// testable place.) No header means a chunked or empty body, which
/// `bodyLimitRefusal` bounds as it arrives.
@Sendable func requestBodyRefusal(contentLength: String?, limit: Int = RequestLimits.maxBodyBytes)
    -> RequestRefusal? {
    guard let contentLength else { return nil }
    guard let declared = Int(contentLength.trimmingCharacters(in: .whitespaces)), declared >= 0 else {
        return RequestRefusal(
            status: .badRequest,
            description: "malformed Content-Length \"\(contentLength)\"")
    }
    guard declared <= limit else {
        return RequestRefusal(
            status: .payloadTooLarge,
            description: "Content-Length \(declared) exceeds this server's request body limit of \(limit) bytes")
    }
    return nil
}

/// The same bound applied as body chunks arrive (chunked transfer carries no
/// `Content-Length`), so a request that declared nothing cannot grow past it.
@Sendable func bodyLimitRefusal(accumulated: Int, incoming: Int, limit: Int = RequestLimits.maxBodyBytes)
    -> RequestRefusal? {
    guard accumulated + incoming > limit else { return nil }
    return RequestRefusal(
        status: .payloadTooLarge,
        description: "request body exceeds this server's limit of \(limit) bytes")
}

/// `model` defaults to the served model's name. Tests pass it explicitly:
/// top-level `main.swift` globals are not initialised when the executable
/// module is imported by a test target, so reading `modelName` there yields a
/// null object that `JSONSerialization` rejects.
@Sendable func completionPayload(
    id: String, text: String?, delta: String?, finish: String?,
    usage: (prompt: Int, completion: Int)? = nil,
    model: String = modelName
) -> [String: Any] {
    var choice: [String: Any] = ["index": 0]
    if let text { choice["message"] = ["role": "assistant", "content": text] }
    if let delta { choice["delta"] = ["content": delta] }
    if delta != nil {
        // OpenAI streaming choices carry the key on every chunk; it is null
        // until the terminal chunk supplies "stop" or "length".
        if let finish {
            choice["finish_reason"] = finish
        } else {
            choice["finish_reason"] = NSNull()
        }
    } else if let finish {
        choice["finish_reason"] = finish
    }
    var payload: [String: Any] = [
        "id": id,
        "object": delta != nil ? "chat.completion.chunk" : "chat.completion",
        "created": Int(Date().timeIntervalSince1970),
        "model": model,
        "choices": [choice],
    ]
    if let usage {
        payload["usage"] = [
            "prompt_tokens": usage.prompt,
            "completion_tokens": usage.completion,
            "total_tokens": usage.prompt + usage.completion,
        ]
    }
    return payload
}

@Sendable func openAIFinishReason(_ reason: GenerationFinishReason?) -> String? {
    switch reason {
    case .length: return "length"
    case .stop: return "stop"
    // Cancellation normally coincides with an inactive channel and therefore
    // writes no terminal chunk. If it is observed on an active channel, omit
    // the field instead of inventing a non-standard OpenAI finish reason.
    case .cancelled, nil: return nil
    }
}

final class HTTPHandler: ChannelInboundHandler {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private var requestHead: HTTPRequestHead?
    private var body = ByteBuffer()
    /// Set once a request has been refused from its head or its body size:
    /// the rest of that request's parts are dropped, the refusal is on the
    /// wire and the connection closes behind it.
    private var refused = false
    private let generationOwner = ConnectionGenerationOwner()

    func channelInactive(context: ChannelHandlerContext) {
        generationOwner.cancelAll()
        context.fireChannelInactive()
    }

    func errorCaught(context: ChannelHandlerContext, error: Swift.Error) {
        generationOwner.cancelAll()
        context.close(promise: nil)
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        switch unwrapInboundIn(data) {
        case .head(let head):
            // Once refused, the connection is closing: nothing further on
            // it is read, including a pipelined request behind the refusal.
            guard !refused else { return }
            body = ByteBuffer()
            if let refusal = requestBodyRefusal(contentLength: head.headers.first(name: "Content-Length")) {
                requestHead = nil
                refuse(context, refusal)
                return
            }
            requestHead = head
        case .body(var chunk):
            guard !refused else { return }
            if let refusal = bodyLimitRefusal(accumulated: body.readableBytes, incoming: chunk.readableBytes) {
                requestHead = nil
                body = ByteBuffer()
                refuse(context, refusal)
                return
            }
            body.writeBuffer(&chunk)
        case .end:
            guard !refused, let head = requestHead else { return }
            route(context: context, head: head, body: body)
            requestHead = nil
        }
    }

    /// Answers a refusal decided before the body was read (or while it was
    /// arriving) and closes the connection once the answer has been flushed:
    /// the client may still be sending the body it was refused for, and
    /// reading it to the end only to discard it is what the limit prevents.
    private func refuse(_ context: ChannelHandlerContext, _ refusal: RequestRefusal) {
        refused = true
        respondJSON(context, status: refusal.status,
                    data: jsonData(["error": refusal.description]), closeAfterFlush: true)
    }

    private func route(context: ChannelHandlerContext, head: HTTPRequestHead, body: ByteBuffer) {
        switch (head.method, head.uri) {
        case (.GET, "/v1/models"):
            respondJSON(context, status: .ok, data: jsonData([
                "object": "list",
                "data": [["id": modelName, "object": "model", "owned_by": "swiftlet"]],
            ]))
        case (.POST, "/v1/chat/completions"):
            handleChat(context: context, body: body)
        default:
            respondJSON(context, status: .notFound, data: jsonData(["error": "not found"]))
        }
    }

    private func handleChat(context: ChannelHandlerContext, body: ByteBuffer) {
        let bytes: [UInt8] = body.getBytes(at: 0, length: body.readableBytes) ?? []
        let data = Data(bytes)
        let request: ChatRequest
        do {
            request = try JSONDecoder().decode(ChatRequest.self, from: data)
        } catch let part as ChatContent.UnsupportedContentPart {
            // Named refusal: the body was well formed, the part is one this
            // server does not serve.
            respondJSON(context, status: .badRequest, data: jsonData(["error": part.description]))
            return
        } catch {
            respondJSON(context, status: .badRequest, data: jsonData(["error": "malformed request"]))
            return
        }
        let messages = request.messages.map { ["role": $0.role, "content": $0.content.text] }
        let maxNew: Int
        do {
            maxNew = try requestedMaxNew(request)
        } catch let refusal as RequestRefusal {
            // Decided here, before the prompt is rendered or queued behind
            // another generation, so a streaming request gets a real 400
            // rather than an error event after a 200 head.
            respondJSON(context, status: refusal.status, data: jsonData(["error": refusal.description]))
            return
        } catch {
            respondJSON(context, status: .badRequest, data: jsonData(["error": "malformed request"]))
            return
        }
        let streaming = request.stream ?? false
        let id = "chatcmpl-\(UUID().uuidString.prefix(8))"
        let eventLoop = context.eventLoop
        let channel = context.channel
        let cancellation = GenerationCancellation()
        let owner = generationOwner
        owner.register(id: id, cancellation: cancellation)

        /// Every chat write is observed. A failed outbound future cancels this
        /// request's token (never a neighboring connection/request), tears down
        /// its heartbeat ownership, and closes the failed channel. The final
        /// successful write is what releases normal ownership.
        @Sendable func observeWrite(
            _ future: EventLoopFuture<Void>,
            terminal: Bool = false
        ) {
            future.whenComplete { result in
                switch result {
                case .success:
                    if terminal { owner.finish(id: id) }
                case .failure:
                    cancellation.cancel()
                    owner.failWrite(id: id)
                    if channel.isActive { channel.close(promise: nil) }
                }
            }
        }

        let options: SwiftletSession.GenerationOptions = {
            var o = SwiftletSession.GenerationOptions()
            if let t = request.temperature {
                if t <= 0 { o = .greedy } else { o.temperature = t }
            }
            if let p = request.top_p { o.topP = p }
            if let p = request.presence_penalty { o.presencePenalty = p }
            if let p = request.frequency_penalty { o.frequencyPenalty = p }
            if let n = request.no_repeat_ngram { o.noRepeatNGram = max(0, n) }
            if let m = request.min_tokens { o.minNew = max(0, m) }
            o.stopSequences = request.stop?.values ?? []
            return o
        }()

        // Agent clients (OpenCode etc.) send multi-thousand-token prompts, so
        // prefill can run minutes with no output. SSE comment heartbeats keep
        // the idle connection from being dropped; parsers ignore them.
        if streaming {
            var head = HTTPResponseHead(version: .http1_1, status: .ok)
            head.headers.add(name: "Content-Type", value: "text/event-stream")
            head.headers.add(name: "Cache-Control", value: "no-cache")
            head.headers.add(name: "Transfer-Encoding", value: "chunked")
            let headPromise = eventLoop.makePromise(of: Void.self)
            observeWrite(headPromise.futureResult)
            context.writeAndFlush(wrapOutboundOut(.head(head)), promise: headPromise)
            let heartbeat = eventLoop.scheduleRepeatedTask(
                initialDelay: .seconds(15), delay: .seconds(15)
            ) { _ in
                guard channel.isActive, !cancellation.isCancelled else { return }
                var buf = channel.allocator.buffer(capacity: 16)
                buf.writeString(": ping\n\n")
                observeWrite(channel.writeAndFlush(
                    HTTPServerResponsePart.body(.byteBuffer(buf))
                ))
            }
            owner.attachHeartbeat(id: id, heartbeat: heartbeat)
        }

        @Sendable func writeSSE(_ obj: [String: Any]) {
            let payload = "data: " + (String(data: jsonData(obj), encoding: .utf8) ?? "{}") + "\n\n"
            eventLoop.execute {
                guard channel.isActive, !cancellation.isCancelled else {
                    if !channel.isActive {
                        cancellation.cancel()
                        owner.failWrite(id: id)
                    }
                    return
                }
                var buf = channel.allocator.buffer(capacity: payload.utf8.count)
                buf.writeString(payload)
                observeWrite(channel.writeAndFlush(
                    HTTPServerResponsePart.body(.byteBuffer(buf))
                ))
            }
        }

        generationQueue.async {
            // The channel may have disappeared while this request waited
            // behind another generation. Do not even construct a stream in
            // that case: prompt preparation and prefill must never start.
            guard !cancellation.isCancelled else {
                eventLoop.execute { owner.finish(id: id) }
                return
            }
            let done = DispatchSemaphore(value: 0)
            Task {
                defer { done.signal() }
                do {
                    var fullText = ""
                    for try await delta in session.streamChat(
                        messages: messages, maxNew: maxNew, options: options,
                        cancellation: cancellation
                    ) {
                        fullText += delta
                        if streaming {
                            writeSSE(completionPayload(id: id, text: nil, delta: delta, finish: nil))
                        }
                    }
                    let m = session.lastMetrics
                    FileHandle.standardError.write(Data(String(
                        format: "[%@] %d prompt + %d generated, ttft %.1fs, %.2f tok/s\n",
                        id as NSString, m.promptTokens, m.generatedTokens,
                        m.timeToFirstToken, m.tokensPerSecond
                    ).utf8))

                    eventLoop.execute {
                        guard channel.isActive, m.finishReason != .cancelled else {
                            cancellation.cancel()
                            owner.failWrite(id: id)
                            return
                        }
                        owner.stopHeartbeat(id: id)
                        let finishReason = openAIFinishReason(m.finishReason)
                        if streaming {
                            // The finish chunk, [DONE], and the HTTP end must
                            // land in this order on the wire. Write them here
                            // directly: routing the finish chunk through
                            // writeSSE would re-enqueue it on the event loop
                            // and it would land after .end and be dropped
                            // (strict clients then wait for it forever).
                            let finish = jsonData(completionPayload(
                                id: id, text: nil, delta: "", finish: finishReason,
                                usage: (m.promptTokens, m.generatedTokens)))
                            let tail = "data: " + (String(data: finish, encoding: .utf8) ?? "{}")
                                + "\n\ndata: [DONE]\n\n"
                            var buf = channel.allocator.buffer(capacity: tail.utf8.count)
                            buf.writeString(tail)
                            observeWrite(channel.write(
                                HTTPServerResponsePart.body(.byteBuffer(buf))
                            ))
                            observeWrite(
                                channel.writeAndFlush(HTTPServerResponsePart.end(nil)),
                                terminal: true
                            )
                        } else {
                            let data = jsonData(completionPayload(
                                id: id, text: fullText, delta: nil, finish: finishReason,
                                usage: (m.promptTokens, m.generatedTokens)))
                            var head = HTTPResponseHead(version: .http1_1, status: .ok)
                            head.headers.add(name: "Content-Type", value: "application/json")
                            head.headers.add(name: "Content-Length", value: String(data.count))
                            var buf = channel.allocator.buffer(capacity: data.count)
                            buf.writeBytes(data)
                            observeWrite(channel.write(HTTPServerResponsePart.head(head)))
                            observeWrite(channel.write(
                                HTTPServerResponsePart.body(.byteBuffer(buf))
                            ))
                            observeWrite(
                                channel.writeAndFlush(HTTPServerResponsePart.end(nil)),
                                terminal: true
                            )
                        }
                    }
                } catch {
                    let message = (error as? LocalizedError)?.errorDescription
                        ?? String(describing: error)
                    let badRequest = error is ContextWindowError
                    eventLoop.execute {
                        guard channel.isActive else {
                            cancellation.cancel()
                            owner.failWrite(id: id)
                            return
                        }
                        owner.stopHeartbeat(id: id)
                        if streaming {
                            // The 200 + SSE head is already on the wire; a
                            // second head would be a protocol error. Report
                            // the failure as an SSE event and end the stream.
                            let payload = "data: " + (String(
                                data: jsonData(["error": message]), encoding: .utf8
                            ) ?? "{}") + "\n\ndata: [DONE]\n\n"
                            var buf = channel.allocator.buffer(capacity: payload.utf8.count)
                            buf.writeString(payload)
                            observeWrite(channel.write(
                                HTTPServerResponsePart.body(.byteBuffer(buf))
                            ))
                            observeWrite(
                                channel.writeAndFlush(HTTPServerResponsePart.end(nil)),
                                terminal: true
                            )
                        } else {
                            let data = jsonData(["error": message])
                            let status: HTTPResponseStatus = badRequest
                                ? .badRequest : .internalServerError
                            var head = HTTPResponseHead(version: .http1_1, status: status)
                            head.headers.add(name: "Content-Type", value: "application/json")
                            head.headers.add(name: "Content-Length", value: String(data.count))
                            var buf = channel.allocator.buffer(capacity: data.count)
                            buf.writeBytes(data)
                            observeWrite(channel.write(HTTPServerResponsePart.head(head)))
                            observeWrite(channel.write(
                                HTTPServerResponsePart.body(.byteBuffer(buf))
                            ))
                            observeWrite(
                                channel.writeAndFlush(HTTPServerResponsePart.end(nil)),
                                terminal: true
                            )
                        }
                    }
                }
            }
            done.wait()
        }
    }

    private func respondJSON(_ context: ChannelHandlerContext, status: HTTPResponseStatus, data: Data,
                             closeAfterFlush: Bool = false) {
        var head = HTTPResponseHead(version: .http1_1, status: status)
        head.headers.add(name: "Content-Type", value: "application/json")
        head.headers.add(name: "Content-Length", value: String(data.count))
        if closeAfterFlush { head.headers.add(name: "Connection", value: "close") }
        var buf = context.channel.allocator.buffer(capacity: data.count)
        buf.writeBytes(data)
        context.write(wrapOutboundOut(.head(head)), promise: nil)
        context.write(wrapOutboundOut(.body(.byteBuffer(buf))), promise: nil)
        if closeAfterFlush {
            let channel = context.channel
            context.writeAndFlush(wrapOutboundOut(.end(nil))).whenComplete { _ in
                channel.close(promise: nil)
            }
        } else {
            context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: nil)
        }
    }
}

let group = MultiThreadedEventLoopGroup(numberOfThreads: 2)
let bootstrap = ServerBootstrap(group: group)
    .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
    .childChannelInitializer { channel in
        channel.pipeline.configureHTTPServerPipeline().flatMap {
            channel.pipeline.addHandler(HTTPHandler())
        }
    }

let channel = try await bootstrap.bind(host: "127.0.0.1", port: port).get()
print("swiftlet-server listening on http://127.0.0.1:\(port)/v1 (model: \(modelName))")
try await channel.closeFuture.get()
