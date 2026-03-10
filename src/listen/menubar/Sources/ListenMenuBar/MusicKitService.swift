/**
 * MusicKitService — lightweight HTTP server exposing MusicKit to the Bun pipeline.
 *
 * Listens on port 3839. The Bun music skill calls these endpoints
 * instead of using AppleScript + UI scripting for catalog tracks.
 *
 * Endpoints:
 *   POST /api/music/play    { "catalogId": "123" } or { "query": "bohemian rhapsody" }
 *   POST /api/music/search  { "query": "bohemian rhapsody", "limit": 5 }
 *   GET  /api/music/status   → current player state + track info
 *
 * Architecture:
 *   Bun skill does library search first (fast AppleScript).
 *   If not found → hits iTunes Search API (free, fast) → gets catalogId.
 *   Then: POST /api/music/play { catalogId } → MusicKit plays instantly.
 */

import Foundation
import Network
import MusicKit

// MARK: - MusicKit Service

@MainActor
class MusicKitService {
    static let shared = MusicKitService()
    private var listener: NWListener?
    private let port: UInt16 = 3839
    private var authorized = false

    func start() {
        Task {
            await requestAuthorization()
            startServer()
        }
    }

    // MARK: - Authorization

    private func requestAuthorization() async {
        let status = await MusicAuthorization.request()
        authorized = status == .authorized
        print("[musickit] authorization: \(status)")
        if !authorized {
            print("[musickit] ⚠ MusicKit not authorized — catalog playback will fail")
            print("[musickit]   grant access in System Settings → Privacy & Security → Media & Apple Music")
        }
    }

    // MARK: - HTTP Server (NWListener)

    private func startServer() {
        do {
            let params = NWParameters.tcp
            listener = try NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
            listener?.newConnectionHandler = { [weak self] conn in
                Task { @MainActor in
                    self?.handleConnection(conn)
                }
            }
            listener?.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    print("[musickit] server ready on :\(self.port)")
                case .failed(let error):
                    print("[musickit] server failed: \(error)")
                default:
                    break
                }
            }
            listener?.start(queue: .global(qos: .userInitiated))
        } catch {
            print("[musickit] failed to start server: \(error)")
        }
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    // MARK: - Connection handling

    private func handleConnection(_ connection: NWConnection) {
        connection.start(queue: .global(qos: .userInitiated))
        connection.receive(minimumIncompleteLength: 1, maximumLength: 65536) { [weak self] data, _, _, error in
            guard let self = self, let data = data, let request = String(data: data, encoding: .utf8) else {
                connection.cancel()
                return
            }

            // Parse the HTTP request line
            let lines = request.components(separatedBy: "\r\n")
            guard let requestLine = lines.first else {
                self.sendResponse(connection, status: 400, body: #"{"error":"bad request"}"#)
                return
            }

            let parts = requestLine.split(separator: " ")
            guard parts.count >= 2 else {
                self.sendResponse(connection, status: 400, body: #"{"error":"bad request"}"#)
                return
            }

            let method = String(parts[0])
            let path = String(parts[1])

            // Extract JSON body (everything after \r\n\r\n)
            let body: String?
            if let range = request.range(of: "\r\n\r\n") {
                let bodyStr = String(request[range.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                body = bodyStr.isEmpty ? nil : bodyStr
            } else {
                body = nil
            }

            // Route
            switch (method, path) {
            case ("POST", "/api/music/play"):
                self.handlePlay(body: body, connection: connection)

            case ("POST", "/api/music/search"):
                self.handleSearch(body: body, connection: connection)

            case ("GET", "/api/music/status"):
                self.handleStatus(connection: connection)

            default:
                self.sendResponse(connection, status: 404, body: #"{"error":"not found"}"#)
            }
        }
    }

    // MARK: - Play endpoint

    private func handlePlay(body: String?, connection: NWConnection) {
        guard let body = body, let data = body.data(using: .utf8) else {
            sendResponse(connection, status: 400, body: #"{"error":"missing body"}"#)
            return
        }

        struct PlayRequest: Codable {
            var catalogId: String?
            var query: String?
        }

        guard let req = try? JSONDecoder().decode(PlayRequest.self, from: data) else {
            sendResponse(connection, status: 400, body: #"{"error":"invalid json"}"#)
            return
        }

        Task { @MainActor in
            do {
                guard authorized else {
                    self.sendResponse(connection, status: 403, body: #"{"error":"MusicKit not authorized"}"#)
                    return
                }

                var song: Song?

                if let catalogId = req.catalogId {
                    // Direct play by catalog ID (fastest path)
                    let request = MusicCatalogResourceRequest<Song>(
                        matching: \.id,
                        equalTo: MusicItemID(rawValue: catalogId)
                    )
                    let response = try await request.response()
                    song = response.items.first
                } else if let query = req.query {
                    // Search and play (fallback)
                    var searchRequest = MusicCatalogSearchRequest(term: query, types: [Song.self])
                    searchRequest.limit = 1
                    let response = try await searchRequest.response()
                    song = response.songs.first
                }

                guard let song = song else {
                    self.sendResponse(connection, status: 404, body: #"{"error":"song not found"}"#)
                    return
                }

                // Play via ApplicationMusicPlayer (macOS 14+)
                if #available(macOS 14.0, *) {
                    let player = ApplicationMusicPlayer.shared
                    player.queue = ApplicationMusicPlayer.Queue(for: [song])
                    try await player.play()
                } else {
                    self.sendResponse(connection, status: 501, body: #"{"error":"requires macOS 14+"}"#)
                    return
                }

                let result: [String: Any] = [
                    "success": true,
                    "track": song.title,
                    "artist": song.artistName,
                    "album": song.albumTitle ?? "",
                    "catalogId": song.id.rawValue,
                ]
                let jsonData = try JSONSerialization.data(withJSONObject: result)
                self.sendResponse(connection, status: 200, body: String(data: jsonData, encoding: .utf8) ?? "{}")

            } catch {
                print("[musickit] play error: \(error)")
                let errorBody = #"{"error":"\#(error.localizedDescription)"}"#
                self.sendResponse(connection, status: 500, body: errorBody)
            }
        }
    }

    // MARK: - Search endpoint

    private func handleSearch(body: String?, connection: NWConnection) {
        guard let body = body, let data = body.data(using: .utf8) else {
            sendResponse(connection, status: 400, body: #"{"error":"missing body"}"#)
            return
        }

        struct SearchRequest: Codable {
            var query: String
            var limit: Int?
        }

        guard let req = try? JSONDecoder().decode(SearchRequest.self, from: data) else {
            sendResponse(connection, status: 400, body: #"{"error":"invalid json"}"#)
            return
        }

        Task { @MainActor in
            do {
                guard authorized else {
                    self.sendResponse(connection, status: 403, body: #"{"error":"MusicKit not authorized"}"#)
                    return
                }

                var searchRequest = MusicCatalogSearchRequest(term: req.query, types: [Song.self])
                searchRequest.limit = req.limit ?? 5
                let response = try await searchRequest.response()

                let results = response.songs.map { song -> [String: Any] in
                    return [
                        "catalogId": song.id.rawValue,
                        "track": song.title,
                        "artist": song.artistName,
                        "album": song.albumTitle ?? "",
                    ]
                }

                let jsonData = try JSONSerialization.data(withJSONObject: ["results": results])
                self.sendResponse(connection, status: 200, body: String(data: jsonData, encoding: .utf8) ?? "[]")

            } catch {
                print("[musickit] search error: \(error)")
                self.sendResponse(connection, status: 500, body: #"{"error":"\#(error.localizedDescription)"}"#)
            }
        }
    }

    // MARK: - Status endpoint

    private func handleStatus(connection: NWConnection) {
        Task { @MainActor in
            var result: [String: Any] = ["authorized": authorized]

            if #available(macOS 14.0, *) {
                let player = ApplicationMusicPlayer.shared
                let state: String
                switch player.state.playbackStatus {
                case .playing: state = "playing"
                case .paused: state = "paused"
                case .stopped: state = "stopped"
                case .interrupted: state = "interrupted"
                case .seekingForward: state = "seeking_forward"
                case .seekingBackward: state = "seeking_backward"
                @unknown default: state = "unknown"
                }

                result["status"] = state

                if let entry = player.queue.currentEntry {
                    if case let .song(song) = entry.item {
                        result["track"] = song.title
                        result["artist"] = song.artistName
                        result["album"] = song.albumTitle ?? ""
                        result["catalogId"] = song.id.rawValue
                    }
                }
            } else {
                result["status"] = "unavailable"
            }

            if let jsonData = try? JSONSerialization.data(withJSONObject: result) {
                self.sendResponse(connection, status: 200, body: String(data: jsonData, encoding: .utf8) ?? "{}")
            } else {
                self.sendResponse(connection, status: 200, body: #"{"status":"unknown"}"#)
            }
        }
    }

    // MARK: - HTTP response helper

    private func sendResponse(_ connection: NWConnection, status: Int, body: String) {
        let statusText: String
        switch status {
        case 200: statusText = "OK"
        case 400: statusText = "Bad Request"
        case 403: statusText = "Forbidden"
        case 404: statusText = "Not Found"
        case 500: statusText = "Internal Server Error"
        default: statusText = "Unknown"
        }

        let headers = [
            "HTTP/1.1 \(status) \(statusText)",
            "Content-Type: application/json",
            "Content-Length: \(body.utf8.count)",
            "Connection: close",
            "Access-Control-Allow-Origin: *",
            "",
            "",
        ].joined(separator: "\r\n")

        let response = headers + body
        connection.send(content: response.data(using: .utf8), completion: .contentProcessed { _ in
            connection.cancel()
        })
    }
}
