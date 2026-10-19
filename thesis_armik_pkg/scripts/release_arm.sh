#!/usr/bin/env bash
# release_arm.sh -- hand the arm back after an armik session (undoes take_arm.sh).
#
#   ./scripts/release_arm.sh            # starts the service again IF take_arm.sh stopped it
#   ./scripts/release_arm.sh --force    # start it even without take_arm.sh's marker
#   ARM_SERVICE=other.service ./scripts/release_arm.sh
#
# Never starts a service this session did not stop (unless --force), so it is
# safe to run out of habit.
set -euo pipefail

SERVICE="${ARM_SERVICE:-shared-growth-workstation.service}"
MARKER="/tmp/armik_stopped_${SERVICE}"

if [[ ! -e "$MARKER" && "${1:-}" != "--force" ]]; then
    echo "take_arm.sh did not stop $SERVICE in this session -- leaving it alone (--force to start it anyway)"
    exit 0
fi

if systemctl is-active --quiet "$SERVICE"; then
    echo "$SERVICE is already running"
else
    echo "starting $SERVICE ..."
    sudo systemctl start "$SERVICE"
fi
rm -f "$MARKER"

if systemctl is-active --quiet "$SERVICE"; then
    echo "$SERVICE is active again -- the arm is handed back."
else
    echo "!! $SERVICE did not come back up -- check: systemctl status $SERVICE"
    exit 1
fi
