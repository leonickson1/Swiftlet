import Foundation
import Hub
import Tokenizers

/// Why a model failed to open, in the four shapes an operator acts on
/// differently. Every error the loaders throw on the way to a ready session
/// maps onto one of these; `swiftlet-server` logs the kind on one line and
/// exits with the kind's code, so a supervisor can tell "the container is
/// missing or corrupt" from "this host has no Metal device" without parsing
/// the message.
///
/// - `container`: the model directory is absent, incomplete or damaged
///   (no config, no tokenizer, a truncated blob, a bad shard).
/// - `config`: `config.json` is present but refused by name
///   (unknown `model_type`, a quantization mode this engine cannot read).
/// - `resource`: the host cannot give what the model needs
///   (an expert cache budget too small for its working set, a KV
///   allocation refused, the port already taken).
/// - `backend`: the runtime itself failed (no Metal device, kernels
///   missing or not compiling), and any error no loader names.
public enum StartupFailure: Swift.Error, CustomStringConvertible, Equatable, Sendable {
    case container(String)
    case config(String)
    case resource(String)
    case backend(String)

    /// Process exit codes, distinct per kind. 1 stays the generic failure,
    /// 2 the usage error, so none of these can be mistaken for either.
    public var exitCode: Int32 {
        switch self {
        case .container: return 3
        case .config: return 4
        case .resource: return 5
        case .backend: return 6
        }
    }

    public var kindName: String {
        switch self {
        case .container: return "container"
        case .config: return "config"
        case .resource: return "resource"
        case .backend: return "backend"
        }
    }

    /// The underlying error's own description, unchanged.
    public var reason: String {
        switch self {
        case .container(let r), .config(let r), .resource(let r), .backend(let r):
            return r
        }
    }

    public var description: String { "\(kindName): \(reason)" }

    /// An error's text the way the server already reports request failures:
    /// `LocalizedError` first, then whatever `String(describing:)` gives (the
    /// `CustomStringConvertible` loaders describe themselves there).
    public static func describe(_ error: Swift.Error) -> String {
        if let localized = (error as? LocalizedError)?.errorDescription { return localized }
        return String(describing: error)
    }

    /// Maps an error thrown while opening a model onto its kind. The
    /// classification is by type, never by message text, so a loader that
    /// rewords an error keeps its kind. An error no loader names is
    /// `backend`: that is where an unexpected throw from the runtime lands,
    /// and the reason carries the original text either way.
    public static func classify(_ error: Swift.Error) -> StartupFailure {
        if let failure = error as? StartupFailure { return failure }
        let reason = describe(error)
        switch error {
        case is QwenConfig.Error:
            return .config(reason)
        case let checkpoint as Checkpoint.Error:
            switch checkpoint {
            case .malformedConfig, .unsupportedQuantMode, .unsupportedBits:
                return .config(reason)
            case .missingConfig, .missingTensor, .badShape:
                return .container(reason)
            }
        case is Qpack.Error, is SafetensorsFile.Error, is MetalShardStore.Error,
             is Hub.HubClientError, is TokenizerError:
            return .container(reason)
        case is ExpertCache.Error, is QwenMetalModel.RuntimeError:
            return .resource(reason)
        case is MetalEngine.Error:
            return .backend(reason)
        case let cocoa as CocoaError:
            switch cocoa.code {
            case .fileNoSuchFile, .fileReadNoSuchFile, .fileReadCorruptFile,
                 .fileReadNoPermission, .fileReadInvalidFileName, .fileReadUnknown:
                return .container(reason)
            default:
                return .backend(reason)
            }
        case let posix as POSIXError:
            switch posix.code {
            case .ENOMEM: return .resource(reason)
            case .ENOENT, .ENOTDIR, .EACCES: return .container(reason)
            default: return .backend(reason)
            }
        default:
            return .backend(reason)
        }
    }
}
