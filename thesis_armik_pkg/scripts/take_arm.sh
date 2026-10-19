#!/usr/bin/env bash
# take_arm.sh -- free the arm's serial port for an armik session on the shared Jetson.
#
#   ./scripts/take_arm.sh [port]          # default port: /dev/ttyTHS1
#   ARM_SERVICE=other.service ./scripts/take_arm.sh
#
# Another student's robot server (shared-growth-workstation.service) starts at
# boot and keeps the arm's serial port open; while it runs, armik's reads fail
# and streamed moves stutter (see scripts/diag_port.py). This stops it for the
# session and leaves a marker so scripts/release_arm.sh starts it again when you
# are done. Its boot setup is NOT changed -- after a reboot it is back anyway.
set -euo pipefail

SERVICE="${ARM_SERVICE:-shared-growth-workstation.service}"
PORT="${1:-/dev/ttyTHS1}"
MARKER="/tmp/armik_stopped_${SERVICE}"

if systemctl is-active --quiet "$SERVICE"; then
    echo "stopping $SERVICE (it holds the arm's port) ..."
    sudo systemctl stop "$SERVICE"
    touch "$MARKER"
    echo "  stopped -- run ./scripts/release_arm.sh when you are done to start it again"
else
    echo "$SERVICE is not running -- nothing to stop"
fi

sleep 2
if ! command -v fuser >/dev/null 2>&1; then
    echo "fuser not available -- cannot re-check $PORT; run: sudo python3 scripts/diag_port.py"
    exit 0
fi
if sudo fuser "$PORT" >/dev/null 2>&1; then
    echo "!! $PORT is STILL in use:"
    sudo fuser -v "$PORT" || true
    echo "   run: sudo python3 scripts/diag_port.py   (it shows what started it)"
    exit 1
fi
echo "$PORT is free -- the arm is yours."
