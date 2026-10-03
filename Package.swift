// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "Swiftlet",
    // Floor chosen to match embedding apps (Priv AI ships iOS 17+). Nothing
    // in the current runtime needs newer OS: kernels are plain Metal compute,
    // runtime-compiled. Gate any future Metal 4/MPP paths with @available.
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "SwiftletCore", targets: ["SwiftletCore"]),
        .executable(name: "swiftlet", targets: ["SwiftletCLI"]),
        .executable(name: "swiftlet-repack", targets: ["SwiftletRepack"]),
        .executable(name: "swiftlet-server", targets: ["SwiftletServer"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-nio", from: "2.65.0"),
    ],
    targets: [
        .target(
            name: "SwiftletCore",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            resources: [.copy("Kernels.metal.txt")]
        ),
        .executableTarget(name: "SwiftletCLI", dependencies: ["SwiftletCore"]),
        .executableTarget(name: "SwiftletRepack", dependencies: ["SwiftletCore"]),
        .executableTarget(
            name: "SwiftletServer",
            dependencies: [
                "SwiftletCore",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ]
        ),
        .testTarget(name: "SwiftletCoreTests", dependencies: ["SwiftletCore"]),
        .testTarget(
            name: "SwiftletServerTests",
            dependencies: ["SwiftletServer", "SwiftletCore"]
        ),
    ]
)
