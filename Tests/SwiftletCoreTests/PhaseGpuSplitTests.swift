import Foundation
import Testing
@testable import SwiftletCore

/// S3b follow-up: the per-phase GPU/wait split is real only where the device
/// can sample timestamps inside a compute encoder (dispatch-boundary
/// counters). Everywhere else the API must report absence instead of
/// publishing zeros that look like measurements, and os_signpost intervals
/// must cover exactly the scopes the timeline already reports so Instruments
/// can see the phases.
@Suite struct PhaseGpuSplitTests {
    static let fixturesDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures")

    /// The probe is runtime evidence, not an assumption: capabilities are
    /// read from the device, and when stage-boundary sampling is advertised a
    /// real sample pass must have resolved. Probing twice must agree.
    @Test func counterProbeIsStableAndFunctional() throws {
        let engine = try MetalEngine()
        let probe = engine.probeCounterSampling()
        #expect(probe == engine.probeCounterSampling(), "probe not deterministic")
        #expect(!probe.deviceName.isEmpty)
        if probe.atStageBoundary && probe.hasTimestampCounterSet {
            #expect(probe.stageBoundarySampleValid == true,
                    "stage-boundary timestamp sampling advertised but a real sample pass did not resolve")
        } else {
            #expect(probe.stageBoundarySampleValid == nil,
                    "functional stage probe ran without the advertised capability")
        }
    }

    /// The model's split support must be derived from the probe, and an
    /// unsupported verdict must carry a reason naming the device.
    @Test func phaseGpuSplitSupportMatchesProbe() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let model = try QwenMetalModel(modelDir: dir)
        let probe = model.counterSamplingSupport
        switch model.phaseGpuSplitSupport {
        case .dispatchBoundaryCounters:
            #expect(probe.atDispatchBoundary && probe.hasTimestampCounterSet,
                    "split claimed without the capability that makes it real")
        case .unsupported(let reason):
            #expect(!(probe.atDispatchBoundary && probe.hasTimestampCounterSet),
                    "capability present but split reported unsupported")
            #expect(!reason.isEmpty, "unsupported verdict without a reason")
            #expect(reason.contains(probe.deviceName), "reason does not name the device")
        }
    }

    /// The discriminator: with dispatch-boundary counters, per-phase GPU sums
    /// must land near the per-buffer GPU totals; without them, the API must
    /// report nil — absent, not zeros-that-look-real.
    @Test func phaseGpuSecondsDiscriminateBySupport() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let model = try QwenMetalModel(modelDir: dir)
        let state = model.makeQwenContext()
        _ = try model.step([1, 5, 9], context: state)
        let metrics = model.lastStepMetrics
        #expect(metrics.completedWithoutThrow)

        switch model.phaseGpuSplitSupport {
        case .dispatchBoundaryCounters:
            let totals = try #require(
                metrics.phaseGpuSeconds, "split supported but step totals absent")
            #expect(totals.values.allSatisfy { $0 >= 0 }, "negative per-phase GPU time")
            var attributed = 0.0
            var attributedBuffers = 0
            for sample in metrics.commandBufferTimeline {
                guard let phaseGpu = sample.phaseGpuSeconds else { continue }
                attributedBuffers += 1
                let sum = phaseGpu.values.reduce(0, +)
                #expect(phaseGpu.values.allSatisfy { $0 >= 0 },
                        "negative per-phase GPU time in a buffer")
                if let gpu = sample.gpuSeconds {
                    // Counter clock vs command-buffer clock: generous but
                    // discriminating — the phase sum must be the same order
                    // as the buffer total, never wildly above it.
                    #expect(sum <= gpu * 1.5 + 5e-4,
                            "phase GPU sum exceeds the buffer's GPU span")
                }
                attributed += sum
            }
            #expect(attributedBuffers > 0, "split supported but no buffer was attributed")
            let totalSum = totals.values.reduce(0, +)
            #expect(abs(attributed - totalSum) < 1e-6,
                    "step totals diverge from the per-buffer attribution")
            #expect(abs(totalSum - metrics.gpuExecutionSeconds)
                    <= max(0.5 * metrics.gpuExecutionSeconds, 2e-3),
                    "per-phase GPU sum far from the per-buffer GPU total")
        case .unsupported:
            #expect(metrics.phaseGpuSeconds == nil,
                    "unsupported split must be absent, not zeros")
            for sample in metrics.commandBufferTimeline {
                #expect(sample.phaseGpuSeconds == nil,
                        "unsupported split leaked per-buffer phase GPU values")
            }
        }
    }

    /// Signpost intervals mirror the timeline exactly: one step interval, one
    /// per committed command buffer, one per phase scope — so an Instruments
    /// trace lines up with the published metrics. The tally rebuilds per step.
    @Test func signpostIntervalsMirrorTheTimeline() throws {
        let dir = Self.fixturesDir.appendingPathComponent("tiny-model-q4")
        let model = try QwenMetalModel(modelDir: dir)
        let state = model.makeQwenContext()
        _ = try model.step([1, 5, 9], context: state)
        let metrics = model.lastStepMetrics
        let tally = model.lastSignpostTally
        #expect(tally.stepIntervals == 1, "step interval count")
        #expect(tally.commandBufferIntervals == metrics.commandBuffersCommitted,
                "command-buffer intervals diverge from committed buffers")
        let phaseScopes = metrics.commandBufferTimeline.reduce(0) { $0 + $1.phases.count }
        #expect(tally.phaseIntervals == phaseScopes,
                "phase intervals diverge from the timeline's phase scopes")

        _ = try model.step([11], context: state)
        let single = model.lastSignpostTally
        #expect(single.stepIntervals == 1, "tally accumulated across steps")
        #expect(single.commandBufferIntervals
                == model.lastStepMetrics.commandBuffersCommitted,
                "tally accumulated across steps")
    }
}
