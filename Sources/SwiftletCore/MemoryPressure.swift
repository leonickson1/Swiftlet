import Foundation

/// The OS memory-pressure levels a session reacts to, the three events a
/// `DispatchSource` memory-pressure source reports.
public enum MemoryPressureLevel: Sendable, Equatable {
    case warning
    case critical
    case normal

    /// One event can carry several bits; the most severe one decides. Nil
    /// for an event carrying none of the three.
    public init?(event: DispatchSource.MemoryPressureEvent) {
        if event.contains(.critical) {
            self = .critical
        } else if event.contains(.warning) {
            self = .warning
        } else if event.contains(.normal) {
            self = .normal
        } else {
            return nil
        }
    }
}

/// A model whose expert cache can be rebuilt at another budget. The Metal
/// model streams its experts through a bounded cache and conforms; the CPU
/// reference holds no cache and does not, so a session over it treats every
/// pressure level as a no-op.
protocol ExpertCacheResizing: AnyObject {
    /// The budget the cache runs at now, nil when this model streams no
    /// experts (a raw checkpoint mapped whole).
    var expertCacheBudgetBytes: Int? { get }
    /// Replaces the cache with one bounded by `gb` (old slots free at once;
    /// the new cache refills lazily). Never called while a step is running.
    func resizeExpertCache(toGB gb: Double)
}

/// Owns the OS memory-pressure source that drives a session's
/// `handleMemoryPressure(_:)`. Registered for `.warning`, `.critical` and
/// `.normal` -- without the last the budget could never be restored. Keep
/// the monitor alive for as long as the session serves; releasing it, or
/// calling `cancel()`, unregisters the source.
public final class MemoryPressureMonitor: @unchecked Sendable {
    public static var events: DispatchSource.MemoryPressureEvent { [.warning, .critical, .normal] }

    private let source: DispatchSourceMemoryPressure

    public init(
        queue: DispatchQueue = DispatchQueue(label: "swiftlet.memory-pressure", qos: .utility),
        onEvent: @escaping @Sendable (MemoryPressureLevel) -> Void
    ) {
        source = DispatchSource.makeMemoryPressureSource(eventMask: Self.events, queue: queue)
        source.setEventHandler { [weak self] in
            guard let self, let level = MemoryPressureLevel(event: self.source.data) else { return }
            onEvent(level)
        }
        source.activate()
    }

    deinit {
        source.cancel()
    }

    /// The events the source is registered for.
    public var mask: DispatchSource.MemoryPressureEvent { source.mask }

    public var isCancelled: Bool { source.isCancelled }

    public func cancel() {
        source.cancel()
    }
}

extension QwenMetalModel: ExpertCacheResizing {
    var expertCacheBudgetBytes: Int? { expertCache?.budgetBytes }

    func resizeExpertCache(toGB gb: Double) {
        resizeCache(toGB: gb)
    }
}
