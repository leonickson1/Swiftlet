#if os(macOS)
import Darwin
import Foundation
import Metal
import Testing
@testable import SwiftletCore

/// Redmine #52584: the test idea from strata 061c5420 and ds4 381808d.
/// No donor code copied. Keep the process footprint independent of other
/// suites, which Swift Testing may run concurrently in the parent process.
@Suite struct MemoryFlatTests {
    private static let steps = 40
    private static let baselineStep = 7
    private static let boundBytes: UInt64 = 32 * 1_048_576
    private static let childKey = "SWIFTLET_MEMORY_FLAT_CHILD"

    private struct Trace {
        let samples: [UInt64]

        var baseline: UInt64 { samples[MemoryFlatTests.baselineStep] }
        var peak: UInt64 { samples[(MemoryFlatTests.baselineStep + 1)...].max()! }
        var exceedsBound: Bool {
            peak > baseline && peak - baseline > MemoryFlatTests.boundBytes
        }
    }

    /// A retained shared buffer is invisible to ordinary Swift heap counters.
    /// Touch every page so a virtual allocation cannot pass the control.
    private final class LeakingStep {
        private var retained: [MTLBuffer] = []

        func step(device: MTLDevice) throws {
            let buffer = try #require(device.makeBuffer(
                length: 16 * 1_048_576, options: .storageModeShared))
            memset(buffer.contents(), 0x5a, buffer.length)
            retained.append(buffer)
        }
    }

    private final class CacheReference {
        weak var value: ExpertCache?
        init(_ value: ExpertCache?) { self.value = value }
    }

    private static func footprint() throws -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let capacity = Int(count)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: capacity) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        try #require(status == KERN_SUCCESS, "task_info(TASK_VM_INFO) failed: \(status)")
        return info.phys_footprint
    }

    private static func measure(leaking: Bool) throws -> Trace {
        let source = MetalModelTests.fixturesDir.appendingPathComponent("tiny-model-q4")
        let container = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-flat-\(UUID().uuidString).qpack")
        defer { try? FileManager.default.removeItem(at: container) }
        var repacker = QpackRepacker(checkpointDir: source, outputDir: container)
        repacker.log = { _ in }
        try repacker.repack()
        let model = try QwenMetalModel(modelDir: container, cacheBudgetGB: 0.01)
        let stride = try #require(model.expertCache?.stride)
        let smallBytes = 16 * stride
        let largeBytes = 32 * stride
        let gb = 1_073_741_824.0
        model.resizeCache(toGB: Double(largeBytes) / gb)
        let state = QwenCPUModel.DecodeState()
        let leak = LeakingStep()
        var samples: [UInt64] = []
        for step in 0..<steps {
            // Alternate the real pressure/relief replacement path: 20 cycles.
            // Only retain the old cache weakly, so the probe cannot leak it.
            let oldCache = CacheReference(model.expertCache)
            let target = step.isMultiple(of: 2) ? smallBytes : largeBytes
            model.resizeCache(toGB: Double(target) / gb)
            #expect(oldCache.value == nil, "ExpertCache retained after resizeCache at step \(step)")
            #expect(model.expertCache?.budgetBytes == target, "ExpertCache budget at step \(step)")
            let logits = try model.step([1 + step % 100], state: state)
            #expect(logits.count == model.config.vocabSize)
            #expect(model.lastStepMetrics.commandBufferErrors == 0)
            #expect(model.expertCache!.allocatedSlots > 0, "ExpertCache slots were not exercised")
            if leaking { try leak.step(device: model.engine.device) }
            let sample = try footprint()
            samples.append(sample)
            print("MEMORY_FLAT_SAMPLE step=\(step) phys_footprint=\(sample)")
        }
        return withExtendedLifetime(leak) { Trace(samples: samples) }
    }

    /// Re-enter only this test in a fresh process. A suite's .serialized trait
    /// alone would still allow unrelated suites to change phys_footprint.
    private static func inChild(_ test: String, body: () throws -> Void) throws {
        if ProcessInfo.processInfo.environment[childKey] == test {
            try body()
            print("MEMORY_FLAT_CHILD_COMPLETED \(test)")
            return
        }
        let log = FileManager.default.temporaryDirectory
            .appendingPathComponent("memory-flat-child-\(UUID().uuidString).log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: log) }
        let handle = try FileHandle(forWritingTo: log)
        defer { try? handle.close() }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
        // Newer SwiftPM uses a separate swift-testing executable and passes
        // the test bundle as an argument. Preserve those loader arguments.
        var arguments = Array(CommandLine.arguments.dropFirst())
        while let index = arguments.firstIndex(of: "--filter") {
            arguments.remove(at: index)
            if index < arguments.count { arguments.remove(at: index) }
        }
        arguments += ["--filter", "MemoryFlatTests/\(test)"]
        process.arguments = arguments
        print("MEMORY_FLAT_CHILD_COMMAND \(CommandLine.arguments[0]) \(arguments)")
        var environment = ProcessInfo.processInfo.environment
        environment[childKey] = test
        process.environment = environment
        process.standardOutput = handle
        process.standardError = handle
        try process.run()
        process.waitUntilExit()
        let output = try String(contentsOf: log, encoding: .utf8)
        print(output)
        #expect(process.terminationStatus == 0, "memory-flat child failed: \(test)")
        #expect(output.contains("MEMORY_FLAT_CHILD_COMPLETED \(test)"),
                "memory-flat child did not execute \(test)")
    }

    @Test func decodeAndCacheReplacementsStayFlat() throws {
        try Self.inChild("decodeAndCacheReplacementsStayFlat") {
            // RED: deliberately inject the leak to prove this assertion fails.
            let trace = try Self.measure(leaking: true)
            print("MEMORY_FLAT_RESULT baseline=\(trace.baseline) peak=\(trace.peak) bound=\(Self.boundBytes)")
            #expect(!trace.exceedsBound,
                    "phys_footprint grew beyond 32 MiB after step 7: baseline=\(trace.baseline), peak=\(trace.peak)")
        }
    }
}
#endif
