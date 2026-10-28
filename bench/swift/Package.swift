// swift-tools-version: 5.9
import PackageDescription

// Swift bench harness for the phi-redact cross-language perf sweep
// (see bench/run_bench.py). Depends on PhiRedactKit by path; the
// kit tree itself is never modified by the bench. Build in release
// with a scratch path outside the repo, e.g.:
//
//   swift build -c release --package-path bench/swift \
//       --scratch-path /tmp/phi-bench-swift
let package = Package(
    name: "bench-swift",
    dependencies: [
        .package(path: "../../../phi-redact-mac-app-v9/PhiRedactKit"),
    ],
    targets: [
        .executableTarget(
            name: "bench-swift",
            dependencies: [.product(name: "PhiRedactKit", package: "PhiRedactKit")],
            path: "Sources/bench-swift"
        ),
    ]
)
