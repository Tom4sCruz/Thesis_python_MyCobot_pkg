#!/usr/bin/env bash
# laptop_demo.sh -- run ON THE LAPTOP: the cube-exchange demo, driving the arm through the Jetson.
#
#   ./scripts/laptop_demo.sh                 # run the demo (scripts/jetson_bridge.sh must run on the Jetson)
#   ./scripts/laptop_demo.sh --on-hand       # any other args go to demo_no-profiles.py
#   ./scripts/laptop_demo.sh --setup         # one-time: Python 3.12 venv, RealSense udev rules, cable link
#
# Settings (env vars): JETSON_IP (192.168.50.2), BRIDGE_PORT (9000),
# ARMIK_VENV (~/.venv-py/armik312), JETSON_USER (for the scp hint).
set -euo pipefail

JETSON_IP="${JETSON_IP:-192.168.50.2}"
LAPTOP_IP="${LAPTOP_IP:-192.168.50.1}"
BRIDGE_PORT="${BRIDGE_PORT:-9000}"
ARMIK_VENV="${ARMIK_VENV:-$HOME/.venv-py/armik312}"
JETSON_USER="${JETSON_USER:-<user>}"
CON="jetson-link"
UDEV_RULES=/etc/udev/rules.d/99-realsense-libusb.rules
UDEV_URL=https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules

cd "$(dirname "$(readlink -f "$0")")/.."          # the package root (thesis_armik_pkg/)
CALIB=scripts/Vision/base_calib.json

calib_hint() {
    echo "   copy it from the Jetson (the camera must not have moved since calibrate_base.py):"
    echo "     scp $JETSON_USER@$JETSON_IP:<repo>/thesis_armik_pkg/$CALIB $PWD/$CALIB"
}

# =============================================================================
# one-time setup
# =============================================================================
if [[ "${1:-}" == "--setup" ]]; then
    # -- Python 3.12 venv (pyrealsense2 / mediapipe have no 3.14 wheels) --------
    if ! command -v uv >/dev/null; then
        echo "installing uv ..."
        sudo pacman -S --needed uv
    fi
    if [[ ! -x "$ARMIK_VENV/bin/python" ]]; then
        echo "creating $ARMIK_VENV (Python 3.12) ..."
        uv venv "$ARMIK_VENV" --python 3.12
    fi
    echo "installing packages into $ARMIK_VENV ..."
    VIRTUAL_ENV="$ARMIK_VENV" uv pip install numpy pymycobot pyserial opencv-python pyrealsense2 mediapipe
    VIRTUAL_ENV="$ARMIK_VENV" uv pip install -e .

    # -- RealSense access without root ------------------------------------------
    if [[ ! -e "$UDEV_RULES" ]]; then
        echo "installing the RealSense udev rules ..."
        sudo curl -fsSL -o "$UDEV_RULES" "$UDEV_URL"
        sudo udevadm control --reload
        sudo udevadm trigger
        echo "   (re-plug the RealSense once)"
    fi

    # -- the cable link: static address, no gateway (Wi-Fi keeps the internet) --
    if nmcli -t -f NAME con show | grep -qx "$CON"; then
        echo "nmcli connection '$CON' already exists"
    else
        mapfile -t eths < <(nmcli -t -f DEVICE,TYPE device | awk -F: '$2=="ethernet"{print $1}')
        if (( ${#eths[@]} == 0 )); then
            echo "!! no Ethernet adapter found -- plug the USB Ethernet adapter in and run --setup again"
            exit 1
        elif (( ${#eths[@]} == 1 )); then
            dev="${eths[0]}"
        else
            echo "Ethernet adapters: ${eths[*]}"
            read -r -p "Which one goes to the Jetson? " dev
        fi
        echo "creating '$CON' on $dev: $LAPTOP_IP/24 ..."
        nmcli con add type ethernet ifname "$dev" con-name "$CON" \
            ipv4.method manual ipv4.addresses "$LAPTOP_IP/24" ipv6.method disabled
    fi
    nmcli con up "$CON" || echo "!! '$CON' did not come up -- is the cable plugged in?"

    if [[ ! -f "$CALIB" ]]; then
        echo
        echo "!! still missing: $CALIB"
        calib_hint
    fi
    echo
    echo "setup done -- run: $0"
    exit 0
fi

# =============================================================================
# run the demo
# =============================================================================
if [[ ! -x "$ARMIK_VENV/bin/python" ]]; then
    echo "!! no Python venv at $ARMIK_VENV -- run: $0 --setup"
    exit 1
fi
# shellcheck disable=SC1091
source "$ARMIK_VENV/bin/activate"

if [[ ! -f "$CALIB" ]]; then
    echo "!! $CALIB is missing (the camera -> robot correction)"
    calib_hint
    exit 1
fi

if [[ "$JETSON_IP" != 127.* ]] && ! ip -br addr | grep -qF " $LAPTOP_IP/"; then
    if nmcli -t -f NAME con show | grep -qx "$CON"; then
        echo "bringing the cable link up ($CON) ..."
        nmcli con up "$CON" || { echo "!! '$CON' did not come up -- is the cable plugged in?"; exit 1; }
    else
        echo "!! no '$CON' connection -- run: $0 --setup"
        exit 1
    fi
fi

echo "waiting for the bridge at $JETSON_IP:$BRIDGE_PORT  (on the Jetson: ./scripts/jetson_bridge.sh) ..."
if ! python3 - "$JETSON_IP" "$BRIDGE_PORT" <<'EOF'
import socket, sys, time
host, port = sys.argv[1], int(sys.argv[2])
end = time.time() + 30
while True:
    try:
        socket.create_connection((host, port), timeout=1).close()
        break
    except OSError:
        if time.time() > end:
            sys.exit(1)
        time.sleep(1)
EOF
then
    echo "!! no bridge after 30 s -- is ./scripts/jetson_bridge.sh running on the Jetson?"
    exit 1
fi
sleep 0.5        # the bridge goes back to listening after that probe connection

echo "bridge up -- starting the demo"
exec python3 scripts/Vision/demo_no-profiles.py --port "socket://$JETSON_IP:$BRIDGE_PORT" "$@"
