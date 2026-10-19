#!/usr/bin/env python3
"""
calibrate_mycobot.py - measure how a real myCobot 280 behaves, so a simulator can be tuned to match.

What it measures (all relative to the pose the arm is in when you start, with small motions):
  1. Robot info: library class, firmware/version strings (if available), joint limits read from the robot.
  2. Read latency: how long get_angles() takes and how often it fails.
  3. Send+read loop time: the fastest rate you can run "send a command, read the state" in a loop.
  4. Step response per joint and per speed setting: delay before motion starts, rise time, real speed in
     degrees/second for each 0-100 "speed" value, overshoot, and how often the position feedback updates.
  5. Streamed tracking: the sort of control the policy will do (a new target every tick). For several rates and
     both "fresh modes", it reports the lag between command and motion, the gain, and the tracking error.

Usage (on the machine connected to the arm):
    pip install pymycobot --upgrade
    python calibrate_mycobot.py --port /dev/ttyUSB0          # Linux  (Windows: --port COM3)
    python calibrate_mycobot.py --dry-run --quick --yes      # test the script with a simulated arm, no robot

SAFETY:
  * The arm is only moved a few degrees around the pose it starts in. Put it in a pose where +-AMP degrees
    on any single joint (default 10) cannot hit the table, the board, or a person.
  * Keep a hand near the power switch. Ctrl+C stops the arm immediately (it does not move afterwards).
  * Speed settings tested default to 20/40/60 (never 100). Amplitude is capped at 25 degrees.

Output: a folder calibration_out/<timestamp>/ with summary.json, suggested_sim_params.json and raw CSV logs.
Bring that whole folder back.
"""
import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime

MAX_AMP = 25.0
# myCobot 280 joint ranges (deg), used whenever the robot does not report a usable
# limit -- many firmwares answer get_joint_min/max_angle with pymycobot's error
# value -1, which would otherwise read as the "limit" (-1, -1)
FALLBACK_LIMITS = [(-168.0, 168.0), (-135.0, 135.0), (-150.0, 150.0),
                   (-145.0, 145.0), (-165.0, 165.0), (-175.0, 175.0)]
LIMIT_MARGIN = 3.0


# ------------------------------------------------------------------ robots
class RealRobot:
    def __init__(self, port, baud):
        cls = None
        try:
            from pymycobot import MyCobot280 as cls  # newer pymycobot versions
        except Exception:
            try:
                from pymycobot import MyCobot as cls  # older versions
            except Exception as exc:
                sys.exit("pymycobot is not installed (%s). Run: pip install pymycobot --upgrade" % exc)
        self.mc = cls(port, baud)
        self.name = cls.__name__
        time.sleep(0.5)

    def get_angles(self):
        try:
            a = self.mc.get_angles()
        except Exception:
            return None
        if isinstance(a, (list, tuple)) and len(a) == 6:
            try:
                return [float(x) for x in a]
            except (TypeError, ValueError):
                return None
        return None

    def send_angles(self, angles, speed):
        self.mc.send_angles([float(x) for x in angles], int(speed))

    def stop(self):
        try:
            self.mc.stop()
        except Exception:
            pass

    def set_fresh_mode(self, mode):
        try:
            self.mc.set_fresh_mode(int(mode))
            return True
        except Exception:
            return False

    def joint_limits(self):
        limits = []
        for j in range(1, 7):
            try:
                lo = self.mc.get_joint_min_angle(j)
                hi = self.mc.get_joint_max_angle(j)
                limits.append((float(lo), float(hi)) if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) else None)
            except Exception:
                limits.append(None)
        return limits

    def info(self):
        out = {"class": self.name}
        for fn in ("get_system_version", "get_basic_version", "get_robot_version", "get_atom_version"):
            try:
                out[fn] = getattr(self.mc, fn)()
            except Exception:
                pass
        return out


class FakeRobot:
    """Simulated arm for --dry-run. The numbers are invented; it only exercises this script."""
    name = "FakeRobot"

    def __init__(self):
        self.pos = [0.0, 20.0, -30.0, 10.0, 0.0, 0.0]
        self.cur = None
        self.pending = []
        self.mode = 0
        self.t_last = time.perf_counter()
        self.fb = list(self.pos)
        self.t_fb = self.t_last
        self.dps_per_speed = 1.2
        self.cmd_latency = 0.04
        self.read_delay = 0.010
        self.fb_period = 0.05

    def _done(self):
        return self.cur is None or all(abs(t - p) < 0.3 for t, p in zip(self.cur[0], self.pos))

    def _advance(self, now):
        while self.t_last < now:
            dt = min(0.005, now - self.t_last)
            self.t_last += dt
            act = [p for p in self.pending if p[0] <= self.t_last]
            if act:
                if self.mode == 1:
                    self.cur = (act[-1][1], act[-1][2])
                    self.pending = [p for p in self.pending if p[0] > self.t_last]
                elif self._done():
                    first = act[0]
                    self.cur = (first[1], first[2])
                    self.pending.remove(first)
            if self.cur:
                tgt, speed = self.cur
                vmax = speed * self.dps_per_speed
                for i in range(6):
                    desired = max(-vmax, min(vmax, 6.0 * (tgt[i] - self.pos[i])))
                    self.pos[i] += desired * dt

    def get_angles(self):
        time.sleep(self.read_delay)
        now = time.perf_counter()
        self._advance(now)
        if now - self.t_fb >= self.fb_period:
            self.fb = list(self.pos)
            self.t_fb = now
        return list(self.fb)

    def send_angles(self, angles, speed):
        time.sleep(0.002)
        now = time.perf_counter()
        self._advance(now)
        self.pending.append((now + self.cmd_latency, [float(a) for a in angles], float(speed)))

    def stop(self):
        self.pending = []
        self.cur = None

    def set_fresh_mode(self, mode):
        self.mode = int(mode)
        return True

    def joint_limits(self):
        return [(-168, 168), (-135, 135), (-150, 150), (-145, 145), (-165, 165), (-175, 175)]

    def info(self):
        return {"class": "FakeRobot (dry run)"}


# ------------------------------------------------------------------ helpers
def pct(vals, p):
    s = sorted(vals)
    if not s:
        return None
    k = (len(s) - 1) * p / 100.0
    f = int(math.floor(k))
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(vals):
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": round(statistics.fmean(vals), 3), "median": round(statistics.median(vals), 3),
            "p95": round(pct(vals, 95), 3), "p99": round(pct(vals, 99), 3), "max": round(max(vals), 3)}


def median_or_none(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 4) if vals else None


def read_angles(robot, tries=10):
    for _ in range(tries):
        a = robot.get_angles()
        if a is not None:
            return a
        time.sleep(0.05)
    sys.exit("Could not read joint angles from the robot. Check the port, baud rate, power and firmware.")


def sanitize_limits(raw):
    """Keep a joint's (lo, hi) only if it is plausible: lo < 0 < hi and wide
    enough for the largest test motion. Anything else -- e.g. (-1, -1), what
    pymycobot returns when the firmware does not answer -- becomes None, so the
    myCobot 280 default in FALLBACK_LIMITS is used for that joint."""
    out = []
    for j, lim in enumerate(raw):
        ok = (lim is not None and len(lim) == 2
              and all(isinstance(v, (int, float)) for v in lim)
              and lim[0] < 0.0 < lim[1]
              and lim[1] - lim[0] >= 2 * (MAX_AMP + LIMIT_MARGIN))
        if lim is not None and not ok:
            lo, hi = FALLBACK_LIMITS[j]
            print("J%d: robot returned limits %s, ignoring -> using the myCobot 280 default (%.0f..%.0f)"
                  % (j + 1, tuple(lim), lo, hi))
        out.append(tuple(lim) if ok else None)
    return out


def clamp_to_limits(angles, limits):
    out = []
    for i, a in enumerate(angles):
        lo, hi = limits[i] if limits[i] else FALLBACK_LIMITS[i]
        out.append(max(lo + LIMIT_MARGIN, min(hi - LIMIT_MARGIN, a)))
    return out


def wait_until_at(robot, target, tol=1.5, timeout=10.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        a = robot.get_angles()
        if a is not None and all(abs(x - y) < tol for x, y in zip(a, target)):
            time.sleep(0.3)
            return True
        time.sleep(0.05)
    return False


def go_home(robot, home, speed=25):
    robot.send_angles(home, speed)
    return wait_until_at(robot, home)


# ------------------------------------------------------------------ tests
def test_read(robot, n):
    durs, fails = [], 0
    t_start = time.perf_counter()
    for _ in range(n):
        t0 = time.perf_counter()
        a = robot.get_angles()
        t1 = time.perf_counter()
        if a is None:
            fails += 1
        else:
            durs.append((t1 - t0) * 1000.0)
    total = time.perf_counter() - t_start
    return {"calls": n, "failures": fails, "call_ms": summarize(durs), "max_read_rate_hz": round(n / total, 2)}


def test_send(robot, home, n, speed):
    send_ms, loop_ms = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        robot.send_angles(home, speed)  # same target as the current pose: no real motion
        t1 = time.perf_counter()
        robot.get_angles()
        t2 = time.perf_counter()
        send_ms.append((t1 - t0) * 1000.0)
        loop_ms.append((t2 - t0) * 1000.0)
    mean_loop = statistics.fmean(loop_ms) / 1000.0
    return {"calls": n, "send_ms": summarize(send_ms), "send_plus_read_ms": summarize(loop_ms),
            "max_loop_rate_hz": round(1.0 / mean_loop, 2)}


def crossing(ts, ys, y0, delta, frac):
    prev_t, prev_v = None, None
    for t, y in zip(ts, ys):
        v = (y - y0) / delta
        if v >= frac:
            if prev_t is None or v == prev_v:
                return t
            return prev_t + (t - prev_t) * (frac - prev_v) / (v - prev_v)
        prev_t, prev_v = t, v
    return None


def analyze_step(ts, ys, y0, target):
    if len(ts) < 5:
        return {"ok": False, "reason": "too_few_samples"}
    tail = [y for t, y in zip(ts, ys) if t >= ts[-1] - 0.4] or ys[-3:]
    final = statistics.median(tail)
    delta = final - y0
    commanded = target - y0
    if abs(commanded) < 0.5:
        # already at the target (e.g. the "down" step after an "up" step that did not move)
        return {"ok": False, "reason": "nothing_to_move", "commanded_deg": round(commanded, 2),
                "moved_deg": round(delta, 2)}
    if abs(delta) < max(0.3 * abs(commanded), 0.2):   # never lets delta be ~0 below
        return {"ok": False, "reason": "little_or_no_motion", "moved_deg": round(delta, 2)}
    settled = (max(tail) - min(tail)) < 1.0 and abs(final - target) < 2.0
    t05, t10, t90 = (crossing(ts, ys, y0, delta, f) for f in (0.05, 0.10, 0.90))
    res = {"ok": True, "settled": settled, "moved_deg": round(delta, 2), "samples": len(ts)}
    if t05 is not None:
        res["delay_to_5pct_s"] = round(t05, 4)
    if t10 is not None and t90 is not None and t90 > t10:
        res["rise_10_90_s"] = round(t90 - t10, 4)
        res["mean_speed_dps"] = round(0.8 * abs(delta) / (t90 - t10), 2)
        moving = [(a, b) for (t, a), (t2, b) in zip(zip(ts, ys), zip(ts[1:], ys[1:])) if t10 <= t <= t90]
        dur = t90 - t10
        res["feedback_updates_per_s"] = round(sum(1 for a, b in moving if a != b) / dur, 2) if dur > 0 else None
    peak, j = 0.0, 0
    for i in range(len(ts)):
        while j < len(ts) and ts[j] - ts[i] < 0.1:
            j += 1
        if j >= len(ts):
            break
        peak = max(peak, abs(ys[j] - ys[i]) / (ts[j] - ts[i]))
    res["peak_speed_dps"] = round(peak, 2)
    over = (max(ys) - final) if delta > 0 else (final - min(ys))
    res["overshoot_pct"] = round(max(0.0, over) / abs(delta) * 100.0, 2)
    return res


def poll_joint(robot, j, duration, t0):
    ts, ys = [], []
    while time.perf_counter() - t0 < duration:
        a = robot.get_angles()
        if a is not None:
            ts.append(time.perf_counter() - t0)
            ys.append(a[j])
    return ts, ys


def run_step(robot, home, limits, j, speed, amp, duration, rows):
    results = []
    if not go_home(robot, home):
        print("   warning: could not confirm arm reached home before step test")
    up = clamp_to_limits(home, limits)
    up[j] = clamp_to_limits([home[k] + (amp if k == j else 0.0) for k in range(6)], limits)[j]
    for direction, tgt_pose in (("up", up), ("down", list(home))):
        base = read_angles(robot)
        t0 = time.perf_counter()
        robot.send_angles(tgt_pose, speed)
        ts, ys = poll_joint(robot, j, duration, t0)
        res = analyze_step(ts, ys, base[j], tgt_pose[j])
        res.update({"joint": j + 1, "speed": speed, "direction": direction})
        results.append(res)
        for t, y in zip(ts, ys):
            rows.append([j + 1, speed, direction, round(t, 5), round(y, 4)])
    return results


def stream_analysis(meas, home, joints, amp, f_sig, phases, warmup):
    per_joint = []
    for k, j in enumerate(joints):
        pts = [(t, a[j]) for t, a in meas if t >= warmup]
        if len(pts) < 10:
            continue

        def cmd(t, k=k, j=j):
            return home[j] + amp * math.sin(2 * math.pi * f_sig * t + phases[k])

        best = None
        lag = 0.0
        while lag <= 1.0:
            err = math.sqrt(statistics.fmean((y - cmd(t - lag)) ** 2 for t, y in pts))
            if best is None or err < best[1]:
                best = (lag, err)
            lag += 0.005
        lag_s, rmse = best
        cm = [cmd(t - lag_s) for t, _ in pts]
        ym = [y for _, y in pts]
        mc_, my = statistics.fmean(cm), statistics.fmean(ym)
        var = sum((c - mc_) ** 2 for c in cm)
        gain = sum((c - mc_) * (y - my) for c, y in zip(cm, ym)) / var if var > 0 else None
        raw = math.sqrt(statistics.fmean((y - cmd(t)) ** 2 for t, y in pts))
        per_joint.append({"joint": j + 1, "lag_s": round(lag_s, 4), "gain": round(gain, 3) if gain else None,
                          "rmse_at_best_lag_deg": round(rmse, 3), "rmse_no_lag_deg": round(raw, 3)})
    return per_joint


def run_stream(robot, home, limits, joints, rate, mode, amp, f_sig, duration, speed, rows):
    ok_mode = robot.set_fresh_mode(mode)
    go_home(robot, home)
    period = 1.0 / rate
    phases = [2 * math.pi * k / max(1, len(joints)) for k in range(len(joints))]
    meas, ticks, late = [], [], 0
    t_start = time.perf_counter()
    next_t = t_start
    last_tick = None
    while True:
        now = time.perf_counter()
        el = now - t_start
        if el >= duration:
            break
        cmd = list(home)
        for k, j in enumerate(joints):
            cmd[j] = home[j] + amp * math.sin(2 * math.pi * f_sig * el + phases[k])
        cmd = clamp_to_limits(cmd, limits)
        robot.send_angles(cmd, speed)
        a = robot.get_angles()
        t_read = time.perf_counter() - t_start
        if a is not None:
            meas.append((t_read, a))
            rows.append([rate, mode, round(el, 5), round(t_read, 5)] + [round(x, 3) for x in cmd] + [round(x, 3) for x in a])
        if last_tick is not None:
            ticks.append(now - last_tick)
        last_tick = now
        next_t += period
        wait = next_t - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        else:
            late += 1
    per_joint = stream_analysis(meas, home, joints, amp, f_sig, phases, warmup=1.0 / f_sig)
    stale = sum(1 for (_, a), (_, b) in zip(meas, meas[1:]) if a == b) / max(1, len(meas) - 1)
    res = {"ok": bool(per_joint), "requested_rate_hz": rate, "fresh_mode": mode, "fresh_mode_accepted": ok_mode,
           "ticks": len(ticks) + 1, "achieved_rate_hz": round((len(ticks) + 1) / duration, 2),
           "tick_interval_ms": summarize([t * 1000 for t in ticks]), "late_tick_fraction": round(late / max(1, len(ticks) + 1), 3),
           "stale_read_fraction": round(stale, 3), "per_joint": per_joint}
    if per_joint:
        res["median_lag_s"] = median_or_none([p["lag_s"] for p in per_joint])
        res["median_gain"] = median_or_none([p["gain"] for p in per_joint])
        res["median_rmse_deg"] = median_or_none([p["rmse_at_best_lag_deg"] for p in per_joint])
        res["rmse_over_amp"] = round(res["median_rmse_deg"] / amp, 3)
    return res


def make_suggestions(summary, amp):
    out = {"note": "Starting points only. Check them against summary.json before using them in the simulator."}
    loop = summary.get("send_loop", {}).get("max_loop_rate_hz")
    streams = [r for r in summary.get("stream", []) if r.get("ok") and (loop is None or r["requested_rate_hz"] <= 0.8 * loop)]
    if streams:
        best = min(streams, key=lambda r: (r["rmse_over_amp"], -r["requested_rate_hz"]))
        out["control_rate_hz"] = best["requested_rate_hz"]
        out["fresh_mode"] = best["fresh_mode"]
        out["action_latency_s"] = best["median_lag_s"]
        out["streamed_gain"] = best["median_gain"]
        out["streamed_rmse_over_amp"] = best["rmse_over_amp"]
    out["speed_setting_to_deg_per_s_mean"] = summary.get("step_speed_map")
    out["speed_setting_to_deg_per_s_peak"] = summary.get("step_peak_speed_map")
    out["joint_limits_deg"] = summary.get("info", {}).get("joint_limits")
    return out


# ------------------------------------------------------------------ main
def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Measure a real myCobot 280 so a simulator can be tuned to it.")
    ap.add_argument("--port", help="Serial port, e.g. /dev/ttyUSB0 or COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dry-run", action="store_true", help="Use a simulated arm (no robot needed)")
    ap.add_argument("--quick", action="store_true", help="Shorter run (for testing the script)")
    ap.add_argument("--yes", action="store_true", help="Skip the safety confirmation prompt")
    ap.add_argument("--amp", type=float, default=10.0, help="Test amplitude in degrees (max %.0f)" % MAX_AMP)
    ap.add_argument("--speeds", type=int, nargs="+", default=[20, 40, 60], help="Speed settings (0-100) for step tests")
    ap.add_argument("--rates", type=float, nargs="+", default=[10, 20, 30], help="Streaming rates (Hz) to test")
    ap.add_argument("--stream-speed", type=int, default=60, help="Speed setting used in the streaming test")
    ap.add_argument("--fresh-modes", type=int, nargs="+", default=[0, 1], help="set_fresh_mode values to compare")
    ap.add_argument("--joints", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6], help="Joints to test (1-6)")
    ap.add_argument("--step-duration", type=float, default=3.0, help="Seconds recorded per step")
    ap.add_argument("--stream-duration", type=float, default=8.0)
    ap.add_argument("--out", default="calibration_out")
    ap.add_argument("--skip", nargs="*", default=[], choices=["read", "send", "step", "stream"])
    args = ap.parse_args()

    amp = min(abs(args.amp), MAX_AMP)
    speeds = [max(1, min(80, s)) for s in args.speeds]
    stream_speed = max(1, min(80, args.stream_speed))
    joints = [j - 1 for j in args.joints if 1 <= j <= 6]
    step_duration, stream_duration, n_read, n_send = args.step_duration, args.stream_duration, 200, 100
    rates, fresh_modes = args.rates, args.fresh_modes
    if args.quick:
        joints, speeds, rates = joints[:1], speeds[1:2] or speeds[:1], rates[1:2] or rates[:1]
        step_duration, stream_duration, n_read, n_send = 1.5, 4.0, 60, 40

    if args.dry_run:
        robot = FakeRobot()
    else:
        port = args.port
        if not port:
            try:
                from serial.tools import list_ports
                ports = [p.device for p in list_ports.comports()]
            except Exception:
                ports = []
            if len(ports) == 1:
                port = ports[0]
            else:
                sys.exit("Pass --port explicitly. Serial ports found: %s" % (ports or "none"))
        robot = RealRobot(port, args.baud)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, stamp)
    os.makedirs(out_dir, exist_ok=True)

    limits = sanitize_limits(robot.joint_limits())
    home = read_angles(robot)
    info = {"timestamp": stamp, "dry_run": args.dry_run, "robot": robot.info(), "home_angles_deg": home,
            "joint_limits": limits, "joint_limits_source": "robot" if all(limits) else "partly missing, fallback used",
            "settings": {"amp_deg": amp, "speeds": speeds, "rates": rates, "fresh_modes": fresh_modes,
                         "stream_speed": stream_speed, "joints": [j + 1 for j in joints]}}
    info["joint_limits"] = [list(l) if l else None for l in limits]

    safe_joints = []
    for j in joints:
        lo, hi = limits[j] if limits[j] else FALLBACK_LIMITS[j]
        if home[j] - amp - LIMIT_MARGIN >= lo and home[j] + amp + LIMIT_MARGIN <= hi:
            safe_joints.append(j)
        else:
            print("Skipping joint %d: +-%.0f deg from the current pose would exceed its limits (%.0f..%.0f)." % (j + 1, amp, lo, hi))
    joints = safe_joints
    if not joints:
        sys.exit("No joint can move +-%.0f deg from this pose. Move the arm to a more central pose and retry." % amp)

    print("\nStarting pose (deg):", [round(a, 1) for a in home])
    print("Each tested joint moves up to +-%.0f deg around this pose. Speeds: %s." % (amp, speeds))
    print("Make sure nothing (table, board, person) is within reach of that motion. Ctrl+C stops the arm.")
    if not args.yes:
        input("Press Enter to start... ")

    summary = {"info": info}
    interrupted = False
    try:
        if "read" not in args.skip:
            print("\n[1/4] Read latency ...")
            summary["read"] = test_read(robot, n_read)
            print("      median %.1f ms, p95 %.1f ms, failures %d" % (summary["read"]["call_ms"].get("median", float("nan")),
                  summary["read"]["call_ms"].get("p95", float("nan")), summary["read"]["failures"]))
        if "send" not in args.skip:
            print("[2/4] Send+read loop speed ...")
            summary["send_loop"] = test_send(robot, home, n_send, speeds[0])
            print("      max loop rate about %.1f Hz" % summary["send_loop"]["max_loop_rate_hz"])
        if "step" not in args.skip:
            print("[3/4] Step response (%d joints x %d speeds x 2 directions) ..." % (len(joints), len(speeds)))
            rows, results = [], []
            for j in joints:
                for s in speeds:
                    r = run_step(robot, home, limits, j, s, amp, step_duration, rows)
                    results += r
                    for x in r:
                        msg = ("delay %.3fs, %.0f deg/s" % (x.get("delay_to_5pct_s", float("nan")), x.get("mean_speed_dps", float("nan")))
                               if x.get("ok") else x.get("reason"))
                        print("      J%d speed %d %s: %s%s" % (x["joint"], s, x["direction"], msg,
                              "" if x.get("settled", True) else "  (not settled: raise --step-duration)"))
            summary["step"] = results
            speed_map = {}
            for s in speeds:
                v = [x.get("mean_speed_dps") for x in results if x.get("ok") and x["speed"] == s]
                speed_map[str(s)] = median_or_none(v)
            summary["step_speed_map"] = speed_map
            # Peak speed is the better estimate of the servo's velocity cap; mean speed is lower on short steps.
            summary["step_peak_speed_map"] = {str(s): median_or_none([x.get("peak_speed_dps") for x in results if x.get("ok") and x["speed"] == s])
                                              for s in speeds}
            summary["step_median_delay_s"] = median_or_none([x.get("delay_to_5pct_s") for x in results if x.get("ok")])
            save_csv(os.path.join(out_dir, "step_response.csv"), ["joint", "speed", "direction", "t_s", "angle_deg"], rows)
        if "stream" not in args.skip:
            print("[4/4] Streamed tracking (%d rates x %d modes) ..." % (len(rates), len(fresh_modes)))
            rows, results = [], []
            for mode in fresh_modes:
                for rate in rates:
                    r = run_stream(robot, home, limits, joints, rate, mode, amp, 0.4, stream_duration, stream_speed, rows)
                    results.append(r)
                    print("      mode %d @ %g Hz: achieved %.1f Hz, lag %s s, rmse/amp %s, stale reads %.0f%%" % (
                        mode, rate, r["achieved_rate_hz"], r.get("median_lag_s"), r.get("rmse_over_amp"), r["stale_read_fraction"] * 100))
            summary["stream"] = results
            header = ["rate_hz", "fresh_mode", "t_send_s", "t_read_s"] + ["cmd_j%d" % i for i in range(1, 7)] + ["meas_j%d" % i for i in range(1, 7)]
            save_csv(os.path.join(out_dir, "stream_tracking.csv"), header, rows)
    except KeyboardInterrupt:
        interrupted = True
        robot.stop()
        print("\nInterrupted: arm stopped. Saving what was measured so far.")
    finally:
        if not interrupted:
            try:
                go_home(robot, home)
            except Exception:
                pass
        summary["interrupted"] = interrupted
        save_json(os.path.join(out_dir, "summary.json"), summary)
        save_json(os.path.join(out_dir, "suggested_sim_params.json"), make_suggestions(summary, amp))

    print("\nDone. Results saved in: %s" % out_dir)
    print("Bring back the whole folder (summary.json, suggested_sim_params.json and the CSV logs).")


if __name__ == "__main__":
    main()