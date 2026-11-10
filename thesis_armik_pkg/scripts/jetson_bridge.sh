#!/usr/bin/env bash
# jetson_bridge.sh -- run ON THE JETSON: lend the arm to the laptop over the cable.
#
#   ./scripts/jetson_bridge.sh             # take the arm, bridge it, give it back on exit (Ctrl-C)
#   ./scripts/jetson_bridge.sh --setup     # one-time: static 192.168.50.2/24 on eth0
#   ./scripts/jetson_bridge.sh --host 0.0.0.0   # extra args go to arm_bridge.py
#
# Does, in order: bring the cable link up (nmcli "laptop-link") -> take_arm.sh
# (stops the other student's service, checks /dev/ttyTHS1 is free) ->
# arm_bridge.py (TCP 9000 <-> /dev/ttyTHS1) -> on ANY exit, release_arm.sh.
# The laptop side is scripts/laptop_demo.sh.
set -euo pipefail

JETSON_IP="${JETSON_IP:-192.168.50.2}"
IFACE="${JETSON_IFACE:-eth0}"
CON="laptop-link"

cd "$(dirname "$(readlink -f "$0")")/.."          # the package root (thesis_armik_pkg/)

if [[ "${1:-}" == "--setup" ]]; then
    if nmcli -t -f NAME con show | grep -qx "$CON"; then
        echo "nmcli connection '$CON' already exists:"
        nmcli -f ipv4.addresses,connection.interface-name con show "$CON"
    else
        echo "This adds a static address $JETSON_IP/24 (no gateway) on $IFACE for the cable to the laptop."
        echo "If $IFACE currently connects the Jetson to another network, that connection is replaced"
        echo "while the cable profile is active.  (other interface: JETSON_IFACE=... $0 --setup)"
        read -r -p "Continue? [y/N] " ans
        [[ "$ans" == [yY]* ]] || { echo "aborted"; exit 1; }
        sudo nmcli con add type ethernet ifname "$IFACE" con-name "$CON" \
            ipv4.method manual ipv4.addresses "$JETSON_IP/24" ipv6.method disabled
    fi
    sudo nmcli con up "$CON" || echo "!! could not bring '$CON' up -- is the cable plugged in?"
    ip -br addr show "$IFACE" || true
    echo "setup done -- next time just run: $0"
    exit 0
fi

# 1. the cable link
if ! ip -br addr | grep -qF " $JETSON_IP/"; then
    if nmcli -t -f NAME con show | grep -qx "$CON"; then
        echo "bringing the cable link up ($CON) ..."
        sudo nmcli con up "$CON" || { echo "!! '$CON' did not come up -- is the cable plugged in?"; exit 1; }
    else
        echo "!! no address $JETSON_IP on this Jetson and no '$CON' connection -- run: $0 --setup"
        exit 1
    fi
fi

# 2. take the arm; from here on, always hand it back
cleanup() {
    echo
    echo "handing the arm back ..."
    ./scripts/release_arm.sh || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM
./scripts/take_arm.sh

# 3. bridge until Ctrl-C
echo
echo "on the laptop run:  ./scripts/laptop_demo.sh   (Ctrl-C here when you are done)"
python3 scripts/arm_bridge.py --host "$JETSON_IP" "$@"
