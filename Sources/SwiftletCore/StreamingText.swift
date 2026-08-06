import Foundation

/// Streaming-safe text deltas over an incrementally decoded token sequence.
///
/// BPE tokenizers split multi-byte UTF-8 characters (emoji in particular)
/// across tokens. Decoding a prefix of such a sequence renders the incomplete
/// tail as U+FFFD; emitting that replacement character is wrong twice over:
/// the user sees a garbage glyph, and once the next token completes the
/// character, the emitted text is no longer a prefix of the full decode — a
/// naive `hasPrefix` delta scheme then goes silent for the rest of the turn.
///
/// `delta(printed:decoded:)` holds back any trailing U+FFFD run until the
/// bytes complete, so the emitted prefix always stays a true prefix of every
/// future decode.
public enum StreamingText {
    /// `decoded` with any trailing U+FFFD run removed (an incomplete
    /// multi-byte character mid-stream). U+FFFD elsewhere is preserved.
    public static func stablePrefix(_ decoded: String) -> String {
        var s = Substring(decoded)
        while s.hasSuffix("\u{FFFD}") { s.removeLast() }
        return String(s)
    }

    /// The next emittable delta, or nil when nothing new can be emitted yet.
    /// On success, the caller should append `delta` to its output and replace
    /// its printed-state with `printed`.
    public static func delta(printed: String, decoded: String) -> (delta: String, printed: String)? {
        let stable = stablePrefix(decoded)
        guard stable.hasPrefix(printed), stable.count > printed.count else { return nil }
        return (String(stable.dropFirst(printed.count)), stable)
    }

    /// Delta for the END of a turn: the sequence is final, so a trailing
    /// U+FFFD can never complete and is emitted as-is rather than held back.
    public static func finalDelta(printed: String, decoded: String) -> String? {
        guard decoded.hasPrefix(printed), decoded.count > printed.count else { return nil }
        return String(decoded.dropFirst(printed.count))
    }
}
