#!/usr/bin/env python3
"""
arm_bridge.py -- run ON THE JETSON: relay the arm's serial port over TCP
========================================================================

Lets a faster computer (the laptop) drive the arm: the laptop does vision,
planning and the setpoint stream, and connects with

    --port socket://192.168.50.2:9000

(any armik script: ArmConnection accepts socket:// / tcp:// ports). This
script only copies bytes between that TCP connection and /dev/ttyTHS1, in
both directions, unchanged -- pymycobot's protocol runs end to end.

    ./scripts/take_arm.sh                     # free the port (shared Jetson)
    python3 scripts/arm_bridge.py             # listen on the cable address
    ...                                       # run the script on the laptop
    Ctrl-C, then ./scripts/release_arm.sh

It listens only on --host (default 192.168.50.2, the direct-cable address)
so nobody on another network can drive the arm; --host 0.0.0.0 listens
everywhere. One client at a time; when it disconnects the bridge waits for
the next one, so the laptop script can be restarted freely.

Needs only pyserial (installed with pymycobot).
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import socket
import threading
import time

import serial

DEFAULT_HOST = "192.168.50.2"
DEFAULT_TCP_PORT = 9000
DEFAULT_SERIAL = "/dev/ttyTHS1"
DEFAULT_BAUD = 1000000


def _port_users(port):
    try:
        from armik.connection import _port_users as users
    except Exception:
        return []
    return users(port)


def serve_client(conn: socket.socket, addr, ser: serial.Serial):
    """Pump bytes both ways until either side closes."""
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    ser.reset_input_buffer()          # don't hand the new client stale replies
    stop = threading.Event()
    counts = {"to_arm": 0, "to_client": 0}

    def serial_to_sock():
        try:
            while not stop.is_set():
                data = ser.read(ser.in_waiting or 1)      # blocks up to ser.timeout
                if data:
                    conn.sendall(data)
                    counts["to_client"] += len(data)
        except (OSError, serial.SerialException) as exc:
            if not stop.is_set():
                print(f"  serial -> client stopped: {exc}")
        finally:
            stop.set()

    t = threading.Thread(target=serial_to_sock, daemon=True)
    t.start()
    try:
        while not stop.is_set():
            data = conn.recv(4096)
            if not data:
                break
            ser.write(data)
            counts["to_arm"] += len(data)
    except OSError as exc:
        print(f"  client -> serial stopped: {exc}")
    finally:
        stop.set()
        t.join(timeout=1.0)
        try:
            conn.close()
        except OSError:
            pass
    print(f"client {addr[0]}:{addr[1]} gone  ({counts['to_arm']} B to the arm, "
          f"{counts['to_client']} B back)")


def main():
    ap = argparse.ArgumentParser(description="Relay the arm's serial port over TCP.")
    ap.add_argument("--host", default=DEFAULT_HOST,
                    help=f"address to listen on (default {DEFAULT_HOST}, the cable link; "
                         f"0.0.0.0 = every interface)")
    ap.add_argument("--tcp-port", type=int, default=DEFAULT_TCP_PORT)
    ap.add_argument("--serial", default=DEFAULT_SERIAL)
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--force", action="store_true",
                    help="open the serial port even if another process holds it")
    args = ap.parse_args()
    _sys.stdout.reconfigure(line_buffering=True)      # log lines show up at once, even in a pipe

    others = [] if args.force else _port_users(args.serial)
    if others:
        who = "; ".join(f"PID {pid}: {cmd[:120]}" for pid, cmd in others)
        print(f"{args.serial} is already open by another process ({who}).\n"
              f"Run ./scripts/take_arm.sh first.", file=_sys.stderr)
        return 1

    ser = serial.Serial()
    ser.port = args.serial
    ser.baudrate = args.baud
    ser.timeout = 0.05
    ser.rts = False                   # as pymycobot opens it
    try:
        ser.open()
    except serial.SerialException as exc:
        print(f"could not open {args.serial}: {exc}", file=_sys.stderr)
        return 1

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((args.host, args.tcp_port))
    except OSError as exc:
        print(f"could not listen on {args.host}:{args.tcp_port}: {exc}\n"
              f"(is the cable link up with that address? `ip -br addr`)", file=_sys.stderr)
        ser.close()
        return 1
    srv.listen(1)
    print(f"bridging {args.serial} @ {args.baud} <-> tcp {args.host}:{args.tcp_port}")
    print(f"laptop: --port socket://{args.host}:{args.tcp_port}     (Ctrl-C to stop)")

    try:
        while True:
            conn, addr = srv.accept()
            print(f"client {addr[0]}:{addr[1]} connected  ({time.strftime('%H:%M:%S')})")
            # one client at a time: a second one waits in the backlog until this ends
            serve_client(conn, addr, ser)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        srv.close()
        ser.close()
    return 0


if __name__ == "__main__":
    _sys.exit(main())
