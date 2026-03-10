import SwiftUI
import AppKit
import AVFoundation
import CoreAudio
import MoonshineVoice
import MusicKit

// MARK: - SSE event models (minimal — stats live on the dashboard)
struct WatchlistSSE: Codable {
    let entryId: String
    let type: String
    let patternId: String
    let category: String
    let severity: String
    let trigger: String
}

struct GateSSE: Codable {
    let entryId: String
    let type: String
    let score: Int
    let reason: String
    let latencyMs: Int
}

// MARK: - Audio Device Manager (CoreAudio)

struct AudioInputDevice: Identifiable, Hashable {
    let id: AudioDeviceID
    let name: String
    let uid: String
}

class AudioDeviceManager: ObservableObject {
    @Published var inputDevices: [AudioInputDevice] = []
    @Published var selectedDeviceID: AudioDeviceID = 0

    init() {
        refresh()
    }

    func refresh() {
        inputDevices = Self.listInputDevices()

        // Try to restore saved preference, otherwise use first available
        if let savedUID = UserDefaults.standard.string(forKey: "selectedMicUID"),
           let match = inputDevices.first(where: { $0.uid == savedUID }) {
            selectedDeviceID = match.id
        } else if let first = inputDevices.first {
            selectedDeviceID = first.id
        }
    }

    func select(_ device: AudioInputDevice) {
        selectedDeviceID = device.id
        UserDefaults.standard.set(device.uid, forKey: "selectedMicUID")
    }

    /// Set the input device on an AVAudioEngine's inputNode via CoreAudio.
    func configureEngine(_ engine: AVAudioEngine, deviceID: AudioDeviceID) throws {
        guard let audioUnit = engine.inputNode.audioUnit else {
            throw NSError(domain: "AudioDeviceManager", code: -1,
                          userInfo: [NSLocalizedDescriptionKey: "No audio unit on inputNode"])
        }

        var devID = deviceID
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &devID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )

        guard status == noErr else {
            throw NSError(domain: "AudioDeviceManager", code: Int(status),
                          userInfo: [NSLocalizedDescriptionKey: "Failed to set audio device (OSStatus \(status))"])
        }
    }

    // MARK: - CoreAudio enumeration

    static func listInputDevices() -> [AudioInputDevice] {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var dataSize: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress, 0, nil, &dataSize
        ) == noErr else { return [] }

        let deviceCount = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
        var deviceIDs = [AudioDeviceID](repeating: 0, count: deviceCount)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress, 0, nil, &dataSize, &deviceIDs
        ) == noErr else { return [] }

        return deviceIDs.compactMap { deviceID -> AudioInputDevice? in
            // Check if device has input channels
            guard hasInputChannels(deviceID) else { return nil }

            let name = getDeviceName(deviceID) ?? "Unknown"
            let uid = getDeviceUID(deviceID) ?? "\(deviceID)"
            return AudioInputDevice(id: deviceID, name: name, uid: uid)
        }
    }

    private static func hasInputChannels(_ deviceID: AudioDeviceID) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioDevicePropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )

        var dataSize: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(deviceID, &address, 0, nil, &dataSize) == noErr,
              dataSize > 0 else { return false }

        let bufferListPointer = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: 1)
        defer { bufferListPointer.deallocate() }

        guard AudioObjectGetPropertyData(deviceID, &address, 0, nil, &dataSize, bufferListPointer) == noErr else {
            return false
        }

        let bufferList = UnsafeMutableAudioBufferListPointer(bufferListPointer)
        let totalChannels = bufferList.reduce(0) { $0 + Int($1.mNumberChannels) }
        return totalChannels > 0
    }

    private static func getDeviceName(_ deviceID: AudioDeviceID) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceNameCFString,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var name: CFString = "" as CFString
        var dataSize = UInt32(MemoryLayout<CFString>.size)
        guard AudioObjectGetPropertyData(deviceID, &address, 0, nil, &dataSize, &name) == noErr else {
            return nil
        }
        return name as String
    }

    private static func getDeviceUID(_ deviceID: AudioDeviceID) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var uid: CFString = "" as CFString
        var dataSize = UInt32(MemoryLayout<CFString>.size)
        guard AudioObjectGetPropertyData(deviceID, &address, 0, nil, &dataSize, &uid) == noErr else {
            return nil
        }
        return uid as String
    }
}

// MARK: - Moonshine Transcriber (custom audio engine with device selection)

/// Uses Moonshine Transcriber + Stream directly (bypassing MicTranscriber)
/// so we can select which audio input device to use via CoreAudio.
class TranscriberEngine {
    private var transcriber: Transcriber?
    private var stream: MoonshineVoice.Stream?
    private var audioEngine: AVAudioEngine?
    private var isListening = false

    private let endpoint = "http://localhost:3838/api/transcript"
    private let modelPath: String
    private let modelArch: ModelArch
    private let sampleRate: Double = 16000
    private let channels: Int = 1
    private let bufferSize: AVAudioFrameCount = 1024
    private var lineCount = 0

    var onLineText: ((String) -> Void)?
    var onLineCompleted: ((String, Float) -> Void)?
    var onError: ((String) -> Void)?

    init() {
        let cacheDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        let defaultPath = cacheDir
            .appendingPathComponent("moonshine_voice")
            .appendingPathComponent("download.moonshine.ai/model/medium-streaming-en/quantized")
            .path

        self.modelPath = ProcessInfo.processInfo.environment["MOONSHINE_MODEL_PATH"] ?? defaultPath
        self.modelArch = .mediumStreaming
    }

    func start(deviceID: AudioDeviceID, deviceManager: AudioDeviceManager) {
        do {
            // 1. Request mic permission
            let permissionStatus = AVCaptureDevice.authorizationStatus(for: .audio)
            if permissionStatus == .denied {
                throw NSError(domain: "Moonshine", code: -1,
                              userInfo: [NSLocalizedDescriptionKey: "Microphone permission denied"])
            }
            if permissionStatus == .notDetermined {
                var granted = false
                let sem = DispatchSemaphore(value: 0)
                AVCaptureDevice.requestAccess(for: .audio) { g in
                    granted = g
                    sem.signal()
                }
                sem.wait()
                if !granted {
                    throw NSError(domain: "Moonshine", code: -1,
                                  userInfo: [NSLocalizedDescriptionKey: "Microphone permission denied"])
                }
            }

            // 2. Create Moonshine transcriber + stream
            let t = try Transcriber(modelPath: modelPath, modelArch: modelArch)
            let s = try t.createStream(updateInterval: 0.5)

            // 3. Add event listener
            s.addListener { [weak self] (event: TranscriptEvent) in
                guard let self = self else { return }

                if let completed = event as? LineCompleted {
                    let text = completed.line.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !text.isEmpty else { return }

                    self.lineCount += 1
                    let duration = completed.line.duration

                    DispatchQueue.main.async {
                        self.onLineCompleted?(text, duration)
                    }
                    self.postTranscript(text: text, duration: duration, lineId: completed.line.lineId)

                } else if let changed = event as? LineTextChanged {
                    let text = changed.line.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !text.isEmpty else { return }
                    DispatchQueue.main.async {
                        self.onLineText?(text)
                    }

                } else if let err = event as? TranscriptError {
                    DispatchQueue.main.async {
                        self.onError?(err.error.localizedDescription)
                    }
                }
            }

            // 4. Start the Moonshine stream
            try s.start()

            // 5. Create AVAudioEngine and set the desired input device
            let engine = AVAudioEngine()

            if deviceID != 0 {
                try deviceManager.configureEngine(engine, deviceID: deviceID)
                let deviceName = deviceManager.inputDevices.first(where: { $0.id == deviceID })?.name ?? "\(deviceID)"
                print("[moonshine] using device: \(deviceName) (id=\(deviceID))")
            } else {
                print("[moonshine] using system default input device")
            }

            let inputNode = engine.inputNode
            let inputFormat = inputNode.inputFormat(forBus: 0)

            // 6. Target format: Float32 mono 16kHz
            guard let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: AVAudioChannelCount(channels),
                interleaved: false
            ) else {
                throw NSError(domain: "Moonshine", code: -1,
                              userInfo: [NSLocalizedDescriptionKey: "Failed to create target audio format"])
            }

            let needsConversion =
                inputFormat.sampleRate != targetFormat.sampleRate
                || inputFormat.channelCount != targetFormat.channelCount
                || inputFormat.commonFormat != targetFormat.commonFormat

            let converter: AVAudioConverter? =
                needsConversion ? AVAudioConverter(from: inputFormat, to: targetFormat) : nil

            // 7. Install tap — feed audio to Moonshine stream
            inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) {
                [weak self, weak s] (buffer, _) in
                guard let self = self, let stream = s, self.isListening else { return }

                var audioData: [Float] = []
                var finalSampleRate = self.sampleRate

                if let converter = converter {
                    let capacity = AVAudioFrameCount(
                        Double(buffer.frameLength) * targetFormat.sampleRate / inputFormat.sampleRate)
                    guard let converted = AVAudioPCMBuffer(
                        pcmFormat: targetFormat, frameCapacity: capacity
                    ) else { return }

                    var error: NSError?
                    let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                        outStatus.pointee = .haveData
                        return buffer
                    }
                    converter.convert(to: converted, error: &error, withInputFrom: inputBlock)
                    if error != nil { return }

                    guard let channelData = converted.floatChannelData else { return }
                    let frameLength = Int(converted.frameLength)
                    audioData.append(
                        contentsOf: UnsafeBufferPointer(start: channelData[0], count: frameLength))
                } else {
                    guard let channelData = buffer.floatChannelData else { return }
                    let frameLength = Int(buffer.frameLength)
                    audioData.append(
                        contentsOf: UnsafeBufferPointer(start: channelData[0], count: frameLength))
                    finalSampleRate = inputFormat.sampleRate
                }

                do {
                    try stream.addAudio(audioData, sampleRate: Int32(finalSampleRate))
                } catch {
                    print("[moonshine] audio feed error: \(error.localizedDescription)")
                }
            }

            // 8. Start engine
            try engine.start()

            self.transcriber = t
            self.stream = s
            self.audioEngine = engine
            self.isListening = true

            print("[moonshine] started — model: \(modelPath)")
            print("[moonshine] audio: \(inputFormat.sampleRate)Hz → \(sampleRate)Hz (conversion: \(needsConversion))")

        } catch {
            print("[moonshine] failed to start: \(error)")
            DispatchQueue.main.async {
                self.onError?("Failed to start: \(error.localizedDescription)")
            }
        }
    }

    func stop() {
        isListening = false

        if let engine = audioEngine {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
            audioEngine = nil
        }

        do {
            try stream?.stop()
        } catch {
            print("[moonshine] stream stop error: \(error)")
        }

        stream?.close()
        stream = nil
        transcriber?.close()
        transcriber = nil
    }

    private func postTranscript(text: String, duration: Float, lineId: UInt64) {
        guard let url = URL(string: endpoint) else { return }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.timeoutInterval = 5

        let body: [String: Any] = [
            "text": text,
            "durationSeconds": duration,
            "lineId": lineId,
            "source": "moonshine",
        ]

        guard let jsonData = try? JSONSerialization.data(withJSONObject: body) else { return }
        request.httpBody = jsonData

        URLSession.shared.dataTask(with: request) { _, _, error in
            if let error = error {
                print("[moonshine] POST failed: \(error.localizedDescription)")
            }
        }.resume()
    }
}

// MARK: - App State

@MainActor
class ListenState: ObservableObject {
    enum Status: String {
        case disconnected
        case listening
        case transcribing
        case watchlistHit
        case escalation
    }

    @Published var status: Status = .disconnected
    @Published var serverConnected = false
    @Published var recentTranscripts: [(id: String, text: String, time: String)] = []
    @Published var recentAlerts: [(category: String, trigger: String, time: String)] = []
    @Published var currentPartial: String = ""
    @Published var transcribing = false

    private var sseTask: Task<Void, Never>?
    private var reconnectTask: Task<Void, Never>?
    private var fadeTask: Task<Void, Never>?
    private let endpoint = "http://localhost:3838"

    let engine = TranscriberEngine()
    let deviceManager = AudioDeviceManager()
    let musicKit = MusicKitService.shared

    func startMusicKit() {
        musicKit.start()
    }

    func startTranscribing() {
        guard !transcribing else { return }
        transcribing = true
        status = .transcribing

        engine.onLineText = { [weak self] text in
            self?.currentPartial = text
        }

        engine.onLineCompleted = { [weak self] text, duration in
            guard let self = self else { return }
            self.currentPartial = ""
            let time = self.nowTime()
            let id = "local-\(Int(Date().timeIntervalSince1970 * 1000))"
            self.recentTranscripts.append((id: id, text: text, time: time))
            if self.recentTranscripts.count > 5 {
                self.recentTranscripts.removeFirst()
            }
        }

        engine.onError = { [weak self] msg in
            print("[listen] transcriber error: \(msg)")
            self?.status = .disconnected
            self?.transcribing = false
        }

        let deviceID = deviceManager.selectedDeviceID
        let dm = deviceManager
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.engine.start(deviceID: deviceID, deviceManager: dm)
        }
    }

    func stopTranscribing() {
        engine.stop()
        transcribing = false
        currentPartial = ""
        status = .disconnected
        serverConnected = false
    }

    /// Restart with a different mic
    func switchDevice(_ device: AudioInputDevice) {
        deviceManager.select(device)
        if transcribing {
            stopTranscribing()
            // Brief delay to let audio engine tear down
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
                self?.startTranscribing()
                self?.connectSSE()
            }
        }
    }

    // MARK: - SSE (dashboard feedback)

    func connectSSE() {
        disconnectSSE()
        sseTask = Task { await streamSSE() }
    }

    func disconnectSSE() {
        sseTask?.cancel()
        sseTask = nil
        reconnectTask?.cancel()
        reconnectTask = nil
    }

    private func streamSSE() async {
        guard let url = URL(string: "\(endpoint)/events") else { return }

        var request = URLRequest(url: url)
        request.timeoutInterval = .infinity

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = .infinity
        config.timeoutIntervalForResource = .infinity
        let session = URLSession(configuration: config)

        do {
            let (stream, response) = try await session.bytes(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                serverConnected = false
                scheduleReconnect()
                return
            }

            serverConnected = true
            if status == .disconnected { status = .listening }

            var currentEvent = ""
            var currentData = ""

            for try await line in stream.lines {
                if Task.isCancelled { break }

                if line.hasPrefix("event: ") {
                    currentEvent = String(line.dropFirst(7))
                } else if line.hasPrefix("data: ") {
                    currentData = String(line.dropFirst(6))
                } else if line.isEmpty && !currentEvent.isEmpty {
                    handleSSEEvent(event: currentEvent, data: currentData)
                    currentEvent = ""
                    currentData = ""
                }
            }
        } catch {
            serverConnected = false
            if !Task.isCancelled {
                scheduleReconnect()
            }
        }
    }

    private func scheduleReconnect() {
        reconnectTask = Task {
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            if !Task.isCancelled { connectSSE() }
        }
    }

    private func handleSSEEvent(event: String, data: String) {
        guard let jsonData = data.data(using: .utf8) else { return }
        let decoder = JSONDecoder()

        switch event {
        case "init":
            print("[listen] SSE connected, received init")

        case "stats":
            break // stats live on the dashboard

        case "watchlist":
            do {
                let evt = try decoder.decode(WatchlistSSE.self, from: jsonData)
                status = .watchlistHit
                recentAlerts.insert(
                    (category: evt.category, trigger: evt.trigger, time: nowTime()),
                    at: 0
                )
                if recentAlerts.count > 5 { recentAlerts.removeLast() }
                scheduleFade()
            } catch {
                print("[listen] SSE watchlist decode error: \(error)")
            }

        case "gate":
            do {
                let evt = try decoder.decode(GateSSE.self, from: jsonData)
                if evt.type == "gate.escalation" {
                    status = .escalation
                    scheduleFade()
                }
            } catch {
                print("[listen] SSE gate decode error: \(error)")
            }

        default:
            break
        }
    }

    private func scheduleFade() {
        fadeTask?.cancel()
        fadeTask = Task {
            try? await Task.sleep(nanoseconds: 10_000_000_000)
            if !Task.isCancelled && (status == .watchlistHit || status == .escalation) {
                status = transcribing ? .transcribing : .listening
            }
        }
    }

    func nowTime() -> String {
        let tf = DateFormatter()
        tf.dateFormat = "HH:mm:ss"
        return tf.string(from: Date())
    }
}

// MARK: - Menu Bar UI

@main
struct ListenMenuBarApp: App {
    @StateObject private var state = ListenState()

    var body: some Scene {
        MenuBarExtra {
            MenuContent(state: state)
        } label: {
            MenuBarLabel(status: state.status)
                .onAppear {
                    // MenuBarLabel's onAppear fires on app launch (it's always visible)
                    state.startTranscribing()
                    state.connectSSE()
                    state.startMusicKit()
                }
        }
    }
}

struct MenuBarLabel: View {
    let status: ListenState.Status

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: iconName)
                .foregroundColor(iconColor)
            Text("listen")
                .font(.system(size: 11, weight: .medium, design: .monospaced))
        }
    }

    var iconName: String {
        switch status {
        case .disconnected: return "ear.badge.waveform"
        case .listening: return "waveform"
        case .transcribing: return "waveform.circle.fill"
        case .watchlistHit: return "heart.fill"
        case .escalation: return "bolt.fill"
        }
    }

    var iconColor: Color {
        switch status {
        case .disconnected: return .gray
        case .listening: return .green
        case .transcribing: return .green
        case .watchlistHit: return .red
        case .escalation: return .yellow
        }
    }
}

struct MenuContent: View {
    @ObservedObject var state: ListenState

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Status header
            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                Text(statusLabel)
                    .font(.system(size: 12, weight: .semibold, design: .monospaced))
                Spacer()
            }
            .padding(.bottom, 4)

            // Live partial transcript
            if !state.currentPartial.isEmpty {
                Text(state.currentPartial)
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.green.opacity(0.8))
                    .lineLimit(2)
                    .padding(.vertical, 2)
            }

            Divider()

            // Mic picker
            Text("MICROPHONE")
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundColor(.secondary)
                .padding(.top, 4)

            ForEach(state.deviceManager.inputDevices) { device in
                Button(action: {
                    state.switchDevice(device)
                }) {
                    HStack(spacing: 6) {
                        Image(systemName: device.id == state.deviceManager.selectedDeviceID
                              ? "mic.circle.fill" : "mic.circle")
                            .foregroundColor(device.id == state.deviceManager.selectedDeviceID
                                             ? .green : .secondary)
                            .font(.system(size: 12))
                        Text(device.name)
                            .font(.system(size: 11, design: .monospaced))
                            .lineLimit(1)
                        Spacer()
                        if device.id == state.deviceManager.selectedDeviceID {
                            Image(systemName: "checkmark")
                                .font(.system(size: 10, weight: .semibold))
                                .foregroundColor(.green)
                        }
                    }
                }
                .buttonStyle(.plain)
                .padding(.vertical, 2)
            }

            Divider()

            // Server connection indicator (minimal — stats live on the dashboard)
            if state.status != .disconnected {
                HStack(spacing: 4) {
                    Circle()
                        .fill(state.serverConnected ? Color.green : Color.orange)
                        .frame(width: 6, height: 6)
                    Text(state.serverConnected ? "pipeline connected" : "pipeline offline")
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 2)

                Divider()
            }

            // Recent transcripts
            if !state.recentTranscripts.isEmpty {
                Text("RECENT")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundColor(.secondary)
                    .padding(.top, 4)

                ForEach(state.recentTranscripts.suffix(3), id: \.id) { entry in
                    HStack(alignment: .top, spacing: 6) {
                        Text(entry.time)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(.secondary)
                        Text(truncate(entry.text, 50))
                            .font(.system(size: 11, design: .monospaced))
                            .lineLimit(2)
                    }
                    .padding(.vertical, 1)
                }

                Divider()
            }

            // Recent alerts
            if !state.recentAlerts.isEmpty {
                Text("ALERTS")
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundColor(.red.opacity(0.8))
                    .padding(.top, 4)

                ForEach(Array(state.recentAlerts.prefix(3).enumerated()), id: \.offset) { _, alert in
                    HStack(alignment: .top, spacing: 6) {
                        Text(alert.time)
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundColor(.secondary)
                        Text("\(alert.category): \"\(truncate(alert.trigger, 30))\"")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.red.opacity(0.9))
                            .lineLimit(1)
                    }
                    .padding(.vertical, 1)
                }

                Divider()
            }

            // Actions
            Button("Open Dashboard") {
                if let url = URL(string: "http://localhost:3838") {
                    NSWorkspace.shared.open(url)
                }
            }
            .keyboardShortcut("d")

            Divider()

            if state.transcribing {
                Button("Stop Transcribing") {
                    state.stopTranscribing()
                    state.disconnectSSE()
                }
            } else {
                Button("Start Transcribing") {
                    state.startTranscribing()
                    state.connectSSE()
                }
            }

            Button("Quit") {
                state.stopTranscribing()
                NSApplication.shared.terminate(nil)
            }
            .keyboardShortcut("q")
        }
        .padding(8)
        .frame(width: 280)
    }

    var statusColor: Color {
        switch state.status {
        case .disconnected: return .gray
        case .listening: return .green
        case .transcribing: return .green
        case .watchlistHit: return .red
        case .escalation: return .yellow
        }
    }

    var statusLabel: String {
        switch state.status {
        case .disconnected: return "disconnected"
        case .listening: return "listening"
        case .transcribing: return "transcribing"
        case .watchlistHit: return "watchlist hit"
        case .escalation: return "escalation"
        }
    }

    func truncate(_ s: String, _ max: Int) -> String {
        s.count > max ? String(s.prefix(max - 1)) + "\u{2026}" : s
    }
}
