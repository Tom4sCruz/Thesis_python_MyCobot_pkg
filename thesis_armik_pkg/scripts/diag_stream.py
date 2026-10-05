#!/usr/bin/env python3
"""
STREAM DIAGNOSTIC -- why does the arm sometimes stutter?
========================================================

Measures the serial link and replays ONE small move (J1 swings out and back
from HOME) several ways, so a stutter can be pinned on a cause instead of
guessed at. Nothing here needs cubes; the gripper is opened / closed a little
in the last two moves (--no-gripper skips those).

    python3 scripts/diag_stream.py --mock --yes          # flow check, no hardware
    python3 scripts/diag_stream.py --port /dev/ttyTHS1   # the real measurement

What it reports
---------------
1. ENVIRONMENT  pymycobot version, whether send_angles() accepts _async, the
   serial read timeout, fresh mode.
2. CALL LATENCY (arm still)  how long one call blocks, for get_angles,
   send_angles the way armik calls it today (synchronous: pymycobot waits for
   a reply and RE-SENDS the command if none comes), send_angles(_async=True)
   (write and return), and the gripper command both ways. A healthy 25 Hz
   stream needs every send to take well under 40 ms, every time.
3. THE SAME MOVE, SEVERAL WAYS -- watch the arm during each:
     M1  one firmware command, no streaming  -> the smoothness the hardware can do
     M2  armik's stream as it is today (arm.move_joints)
     M3  the same stream, asynchronous sends
     M4  asynchronous stream with setpoints 5 ms apart (deliberately too fast)
     M5a M3 + gripper commands mid-move, sent the blocking way
     M5b M3 + gripper commands mid-move, written without waiting for a reply
   For each streamed move: time spent inside the send call per tick, the
   longest gap between two sends, late ticks, planned vs actual duration.

Paste the whole output back, plus which of M1..M5b LOOKED smooth.
"""

from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import inspect
import time

import numpy as np

from armik import Arm, config
from armik.connection import dps_to_firmware_speed

HOME = [0.0, 0.0, -90.0, 0.0, 0.0, 0.0]
LATENCY_CALLS = 40
PAUSE_BETWEEN_MOVES_S = 1.5
GRIP_A, GRIP_B = 90, 45          # gripper values (0-100) toggled in M5a / M5b
GRIP_SPEED = 80


def _stats_ms(samples_s):
    a = np.asarray(samples_s, dtype=float) * 1000.0
    if a.size == 0:
        return "n/a"
    return (f"min {a.min():6.1f}  median {np.median(a):6.1f}  "
            f"p95 {np.percentile(a, 95):6.1f}  max {a.max():6.1f}  ms  (n={a.size})")


def _timed(fn, n):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
        time.sleep(0.02)
    return out


def _min_jerk(tau):
    t = np.clip(tau, 0.0, 1.0)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


class Link:
    """The raw pymycobot object + the few call variants being compared."""

    def __init__(self, arm):
        self.arm = arm
        self.mc = arm.conn.raw
        self.lock = arm.conn.lock
        try:
            self.has_async = "_async" in inspect.signature(self.mc.send_angles).parameters
        except (TypeError, ValueError):
            self.has_async = False
        self._genre_grip = None
        if hasattr(self.mc, "_mesg"):
            try:
                from pymycobot.common import ProtocolCode
                self._genre_grip = ProtocolCode.SET_GRIPPER_VALUE
            except Exception:
                self._genre_grip = None

    def send_sync(self, q, speed):
        with self.lock:
            self.mc.send_angles([round(float(a), 2) for a in q], int(speed))

    def send_async(self, q, speed):
        if not self.has_async:
            return self.send_sync(q, speed)
        with self.lock:
            self.mc.send_angles([round(float(a), 2) for a in q], int(speed), _async=True)

    def grip_sync(self, value):
        with self.lock:
            self.mc.set_gripper_value(int(value), GRIP_SPEED)

    @property
    def has_grip_nowait(self):
        return self._genre_grip is not None and self.has_async

    def grip_nowait(self, value):
        if not self.has_grip_nowait:
            return self.grip_sync(value)
        with self.lock:
            self.mc._mesg(self._genre_grip, int(value), GRIP_SPEED, _async=True)

    def drain(self):
        """Throw away replies nobody read (after asynchronous sends)."""
        time.sleep(0.3)
        port = getattr(self.mc, "_serial_port", None)
        if port is not None and hasattr(port, "reset_input_buffer"):
            with self.lock:
                try:
                    port.reset_input_buffer()
                except Exception:
                    pass


def wait_until_at(arm, q_target, timeout_s=15.0, tol=1.5):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            q = arm.get_angles()
        except Exception:
            q = None
        if q is not None and max(abs(a - b) for a, b in zip(q, q_target)) <= tol:
            return time.perf_counter() - t0
        time.sleep(0.1)
    return None


def stream(link, q_from, q_to, duration_s, tick_s, send, events=()):
    """Own min-jerk stream on an absolute-deadline schedule (the same scheme
    as Arm._execute). events: [(fraction_of_move, callable), ...] fired inside
    the loop right after that tick's setpoint. Returns a stats dict."""
    q_from, q_to = np.asarray(q_from, float), np.asarray(q_to, float)
    n = max(2, int(round(duration_s / tick_s)) + 1)
    ts = np.linspace(0.0, duration_s, n)
    wps = q_from[None, :] + (q_to - q_from)[None, :] * _min_jerk(ts / duration_s)[:, None]
    pending = sorted(events, key=lambda e: e[0])
    send_s, sent_at, late, ev_s = [], [], 0, []
    t0 = time.perf_counter()
    prev = wps[0]
    for k in range(1, n):
        deadline = t0 + ts[k]
        now = time.perf_counter()
        dt = ts[k] - ts[k - 1]
        if now < deadline:
            time.sleep(deadline - now)
        elif now > deadline + dt:
            late += 1
        dps = float(np.max(np.abs(wps[k] - prev)) / dt) * config.STREAM_SPEED_GAIN
        speed = dps_to_firmware_speed(max(dps, 1.0)).firmware_speed
        a = time.perf_counter()
        send(wps[k], speed)
        b = time.perf_counter()
        send_s.append(b - a)
        sent_at.append(b - t0)
        prev = wps[k]
        while pending and k / (n - 1) >= pending[0][0]:
            _, fn = pending.pop(0)
            c = time.perf_counter()
            fn()
            ev_s.append(time.perf_counter() - c)
    return {"send_s": send_s, "gaps_s": list(np.diff(sent_at)), "late": late,
            "planned_s": duration_s, "actual_s": time.perf_counter() - t0,
            "setpoints": n - 1, "tick_s": tick_s, "event_s": ev_s}


def merge(a, b):
    out = dict(a)
    for key in ("send_s", "gaps_s", "event_s"):
        out[key] = a[key] + b[key]
    for key in ("late", "planned_s", "actual_s", "setpoints"):
        out[key] = a[key] + b[key]
    return out


def report(name, st):
    print(f"  {name}: {st['setpoints']} setpoints, tick {st['tick_s'] * 1000:.0f} ms, "
          f"planned {st['planned_s']:.2f}s actual {st['actual_s']:.2f}s, LATE ticks {st['late']}")
    print(f"      time inside send : {_stats_ms(st['send_s'])}")
    print(f"      gap between sends: {_stats_ms(st['gaps_s'])}")
    if st["event_s"]:
        print(f"      gripper call     : {_stats_ms(st['event_s'])}")


def banner(text):
    print(f"\n=== {text} ===")
    time.sleep(PAUSE_BETWEEN_MOVES_S)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=config.DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=config.DEFAULT_BAUDRATE)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the safety prompt")
    ap.add_argument("--swing-deg", type=float, default=35.0, help="J1 swing away from HOME")
    ap.add_argument("--move-s", type=float, default=2.0, help="duration of each one-way move")
    ap.add_argument("--no-gripper", action="store_true", help="skip every gripper command")
    args = ap.parse_args()

    if not args.mock and not args.yes:
        print("This moves the arm to HOME, swings J1 a few times and opens/closes the gripper. "
              "Clear the workspace.")
        if input("Type 'go' to continue: ").strip().lower() != "go":
            return 1

    arm = Arm(port=args.port, baudrate=args.baud, mock=args.mock)
    link = Link(arm)
    tick = 1.0 / config.CONTROL_RATE_HZ
    q_a = list(HOME)
    q_b = list(HOME)
    q_b[0] += args.swing_deg
    try:
        if not arm.conn.is_power_on():
            print("powering on...")
            arm.conn.power_on()
            time.sleep(1.5)

        # ---- 1. environment -------------------------------------------------
        print("\n##### 1. ENVIRONMENT #####")
        try:
            import pymycobot
            print(f"  pymycobot version        : {getattr(pymycobot, '__version__', '?')}")
        except Exception as exc:
            print(f"  pymycobot                : not importable here ({exc})")
        print(f"  raw object               : {type(link.mc).__name__}")
        print(f"  send_angles has _async   : {link.has_async}")
        print(f"  gripper no-wait possible : {link.has_grip_nowait}")
        port = getattr(link.mc, "_serial_port", None)
        print(f"  serial read timeout      : {getattr(port, 'timeout', 'n/a')}")
        try:
            print(f"  fresh mode               : {arm.conn.get_fresh_mode()}")
        except Exception as exc:
            print(f"  fresh mode               : could not read ({exc})")
        print(f"  CONTROL_RATE_HZ / GAIN   : {config.CONTROL_RATE_HZ} / {config.STREAM_SPEED_GAIN}")

        print("\nhoming (one firmware command)...")
        link.send_sync(q_a, 30)
        if wait_until_at(arm, q_a) is None:
            print("  !! did not reach HOME within the timeout -- results below may be off")
        time.sleep(0.5)

        # ---- 2. call latency ------------------------------------------------
        print("\n##### 2. CALL LATENCY, ARM STILL #####")
        print(f"  get_angles               : {_stats_ms(_timed(arm.get_angles, LATENCY_CALLS))}")
        print(f"  send_angles  synchronous : "
              f"{_stats_ms(_timed(lambda: link.send_sync(q_a, 30), LATENCY_CALLS))}")
        if link.has_async:
            print(f"  send_angles  _async=True : "
                  f"{_stats_ms(_timed(lambda: link.send_async(q_a, 30), LATENCY_CALLS))}")
            link.drain()
        else:
            print("  send_angles  _async=True : not supported by this pymycobot")
        if not args.no_gripper:
            g0 = arm.get_gripper_value()
            g0 = GRIP_A if g0 is None else int(np.clip(g0, 0, 100))
            print(f"  gripper      synchronous : "
                  f"{_stats_ms(_timed(lambda: link.grip_sync(g0), 12))}")
            if link.has_grip_nowait:
                print(f"  gripper      no wait     : "
                      f"{_stats_ms(_timed(lambda: link.grip_nowait(g0), 12))}")
                link.drain()
            else:
                print("  gripper      no wait     : not supported by this pymycobot")

        # ---- 3. the same move, several ways ---------------------------------
        print("\n##### 3. THE SAME MOVE, SEVERAL WAYS -- watch the arm #####")
        dps = args.swing_deg / args.move_s

        banner("M1  one firmware command each way (no streaming)")
        spd = dps_to_firmware_speed(dps * 1.3).firmware_speed
        for target in (q_b, q_a):
            t = time.perf_counter()
            link.send_sync(target, spd)
            call = time.perf_counter() - t
            took = wait_until_at(arm, target)
            print(f"  M1: send call {call * 1000:.1f} ms, arrived after "
                  f"{'TIMEOUT' if took is None else f'{took:.2f}s'} (firmware speed {spd})")
            time.sleep(0.4)

        banner("M2  armik stream as it is today (arm.move_joints, synchronous sends)")
        tot_late, gaps, dur = 0, [], 0.0
        for target in (q_b, q_a):
            ok = arm.move_joints(target, duration=args.move_s)
            ex = arm.last_execution
            if not ok or ex is None:
                print(f"  M2: move_joints REFUSED -- {arm.last_error}")
                continue
            tot_late += ex.late_deadlines
            gaps += list(np.diff(ex.t_cmd))
            dur += ex.duration_s
            time.sleep(0.4)
        print(f"  M2: planned {2 * args.move_s:.2f}s actual {dur:.2f}s, LATE ticks {tot_late}")
        print(f"      gap between sends: {_stats_ms(gaps)}")

        banner("M3  the same stream, asynchronous sends"
               + ("" if link.has_async else "  [NOT SUPPORTED -> identical to synchronous]"))
        st = stream(link, q_a, q_b, args.move_s, tick, link.send_async)
        time.sleep(0.4)
        st = merge(st, stream(link, q_b, q_a, args.move_s, tick, link.send_async))
        link.drain()
        report("M3", st)

        banner("M4  asynchronous stream, setpoints 5 ms apart (deliberately too fast)")
        st = stream(link, q_a, q_b, args.move_s, 0.005, link.send_async)
        time.sleep(0.4)
        st = merge(st, stream(link, q_b, q_a, args.move_s, 0.005, link.send_async))
        link.drain()
        report("M4", st)

        if not args.no_gripper:
            for name, grip, note in (
                    ("M5a", link.grip_sync, "gripper sent the BLOCKING way"),
                    ("M5b", link.grip_nowait, "gripper written WITHOUT waiting"
                     + ("" if link.has_grip_nowait else "  [NOT SUPPORTED -> blocking]"))):
                banner(f"{name}  M3 + gripper commands mid-move, {note}")
                st = stream(link, q_a, q_b, args.move_s, tick, link.send_async,
                            events=[(0.35, lambda g=grip: g(GRIP_A))])
                time.sleep(0.4)
                st = merge(st, stream(link, q_b, q_a, args.move_s, tick, link.send_async,
                                      events=[(0.35, lambda g=grip: g(GRIP_B))]))
                link.drain()
                report(name, st)

        print("\n##### DONE #####")
        print("Paste everything above, and say which of M1, M2, M3, M4, M5a, M5b LOOKED smooth")
        print("and which stuttered (and, for M5a/M5b, whether the gripper actually moved).")
        return 0

    except KeyboardInterrupt:
        print("\nCtrl+C -- stopping the arm.")
        arm.stop()
        return 1
    finally:
        arm.close()


if __name__ == "__main__":
    _sys.exit(main())
