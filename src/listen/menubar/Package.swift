// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "ListenMenuBar",
    platforms: [.macOS(.v13)],
    dependencies: [
        .package(url: "https://github.com/moonshine-ai/moonshine-swift.git", from: "0.0.49"),
    ],
    targets: [
        .executableTarget(
            name: "ListenMenuBar",
            dependencies: [
                .product(name: "Moonshine", package: "moonshine-swift"),
            ],
            path: "Sources/ListenMenuBar",
            linkerSettings: [
                .linkedLibrary("c++"),
            ]
        ),
    ]
)
