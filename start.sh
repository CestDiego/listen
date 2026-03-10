#!/bin/bash
# start.sh — launch listen (Bun pipeline + Moonshine menu bar app)
#
# Usage:
#   ./start.sh              # start both
#   ./start.sh --build      # build the menu bar app first, then start
#   ./start.sh --stop       # stop everything

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BINARY="$SCRIPT_DIR/bin/ListenMenuBar"
PORT=3838

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[listen]${NC} $1"; }
ok()    { echo -e "${GREEN}[listen]${NC} $1"; }
warn()  { echo -e "${YELLOW}[listen]${NC} $1"; }
err()   { echo -e "${RED}[listen]${NC} $1"; }

# ── Stop mode ───────────────────────────────────────────────────────
if [ "$1" = "--stop" ]; then
    info "stopping..."
    pkill -f "ListenMenuBar" 2>/dev/null && ok "menu bar app stopped" || warn "menu bar app not running"
    lsof -ti :$PORT 2>/dev/null | xargs kill 2>/dev/null && ok "bun server stopped" || warn "bun server not running"
    exit 0
fi

# ── Preflight checks ───────────────────────────────────────────────
info "preflight checks..."

if ! command -v bun &>/dev/null; then
    err "bun not found. Install: curl -fsSL https://bun.sh/install | bash"
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/node_modules" ]; then
    info "installing dependencies..."
    cd "$SCRIPT_DIR" && bun install
fi

# ── Build mode ──────────────────────────────────────────────────────
if [ "$1" = "--build" ]; then
    info "building menu bar app (xcodebuild)..."
    if ! command -v xcodebuild &>/dev/null; then
        err "xcodebuild not found. Install Xcode from the App Store."
        exit 1
    fi
    cd "$SCRIPT_DIR/src/listen/menubar"
    if xcodebuild -scheme ListenMenuBar -destination 'platform=macOS' build 2>&1 | tail -3 | grep -q "BUILD SUCCEEDED"; then
        ok "build succeeded"
        mkdir -p "$SCRIPT_DIR/bin"
        BUILT=$(find ~/Library/Developer/Xcode/DerivedData/menubar-*/Build/Products/Debug/ListenMenuBar -maxdepth 0 2>/dev/null | head -1)
        if [ -n "$BUILT" ]; then
            cp "$BUILT" "$BINARY"
            ok "binary copied to bin/ListenMenuBar"
        else
            err "could not find built binary in DerivedData"
            exit 1
        fi
    else
        err "build failed"
        exit 1
    fi
    cd "$SCRIPT_DIR"
fi

# ── Check binary exists ────────────────────────────────────────────
if [ ! -f "$BINARY" ]; then
    err "menu bar binary not found at $BINARY"
    info "run: ./start.sh --build"
    exit 1
fi

# ── Check Moonshine model ──────────────────────────────────────────
MODEL_DIR="$HOME/Library/Caches/moonshine_voice/download.moonshine.ai/model/medium-streaming-en/quantized"
if [ ! -d "$MODEL_DIR" ]; then
    warn "moonshine model not found at $MODEL_DIR"
    info "install: pip install moonshine-voice && python -m moonshine_voice.download --language en"
    exit 1
fi

# ── Kill old processes ──────────────────────────────────────────────
pkill -f "ListenMenuBar" 2>/dev/null && warn "killed old menu bar app" || true
lsof -ti :$PORT 2>/dev/null | xargs kill 2>/dev/null && warn "killed old server on :$PORT" || true
sleep 0.5

# ── Start Bun pipeline ─────────────────────────────────────────────
info "starting bun pipeline (moonshine mode)..."
cd "$SCRIPT_DIR"
bun run src/listen/index.ts --moonshine &
BUN_PID=$!

# Wait for server to be ready
for i in $(seq 1 10); do
    if curl -s http://localhost:$PORT/api/session >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

if ! curl -s http://localhost:$PORT/api/session >/dev/null 2>&1; then
    err "bun server failed to start on :$PORT"
    kill $BUN_PID 2>/dev/null
    exit 1
fi
ok "bun server ready on :$PORT"

# ── Start menu bar app ─────────────────────────────────────────────
info "starting menu bar app..."
"$BINARY" &
MENUBAR_PID=$!
sleep 2

if kill -0 $MENUBAR_PID 2>/dev/null; then
    ok "menu bar app running (pid $MENUBAR_PID)"
else
    err "menu bar app crashed"
    kill $BUN_PID 2>/dev/null
    exit 1
fi

# ── Ready ───────────────────────────────────────────────────────────
echo ""
ok "listen is running"
info "dashboard:  http://localhost:$PORT"
info "events:     tail -f /tmp/listen-events.jsonl"
info "stop:       ./start.sh --stop"
echo ""

# ── Wait for either process to exit ─────────────────────────────────
cleanup() {
    echo ""
    info "shutting down..."
    kill $MENUBAR_PID 2>/dev/null
    kill $BUN_PID 2>/dev/null
    wait $BUN_PID 2>/dev/null
    wait $MENUBAR_PID 2>/dev/null
    ok "stopped."
}
trap cleanup SIGINT SIGTERM

wait $BUN_PID $MENUBAR_PID 2>/dev/null
cleanup
