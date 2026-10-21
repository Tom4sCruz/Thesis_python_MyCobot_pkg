#!/usr/bin/env python3
"""
calibrate_mycobot_v2.py - measure how a real myCobot 280 behaves, so a simulator can be tuned to it.

Why a v2: the first run (calibrate_mycobot.py) found that "send a command, then read" took about half a second, that the angle
feedback often froze during a move, and that streaming reached only 2-3 Hz. It could not say WHICH call was slow, its speed numbers
came from interpolating across frozen feedback, and it never moved more than 10 degrees. This version:

  * times every call (start AND end), so we can see whether the send or the read blocks;
  * tests sends with NO reads, at the rates the policy needs (10/25/50 Hz), in both fresh modes;
  * tests "send, wait X s, read" to see whether a read waits for the arm to finish the command just sent;
  * discards step traces whose feedback froze (it never interpolates across a freeze) and says how many were discarded;
  * uses bigger steps (default 10 and 30 deg) so the speed cap, not the acceleration, is what gets measured;
  * repeats the step test in a second pose (--pose-b) because gravity load differs from pose to pose;
  * searches lags up to 3 s and flags results that hit the search limit;
  * records the pymycobot version and, optionally, the pose the arm takes when sent all zeros.

Usage (on the machine connected to the arm):
    pip install pymycobot --upgrade
    python calibrate_mycobot_v2.py --port COM3                                  # Windows   (Linux: /dev/ttyUSB0)
    python calibrate_mycobot_v2.py --port COM3 --pose-b 0,30,-30,0,0,0          # + a second, more extended pose (see below)
    python calibrate_mycobot_v2.py --dry-run --quick --yes                      # test the script with a simulated arm
    python calibrate_mycobot_v2.py --selftest                                   # check the analysis code, no arm needed

SAFETY
  * The arm moves only around the pose it starts in: +-AMP degrees on ONE joint at a time (AMP up to 25; up to 60 only with
    --allow-large, which asks for confirmation). Steps go to the side of the joint's range that has more room.
  * --pose-b: the arm is first sent there at a slow speed. Choose a pose with nothing within reach, e.g. forearm roughly
    horizontal, tool pointing away from you. The script prints it and asks for confirmation.
  * --zero-pose sends all joints to 0 (arm upright): needs about 40 cm of free space above the base.
  * Keep a hand near the power switch. Ctrl+C stops the arm. Speed settings are limited to 80.

Output: calibration_out/<timestamp>/ with summary.json, suggested_sim_params.json and raw CSV logs. Bring back the whole folder.
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime

MAX_AMP, MAX_AMP_LARGE = 25.0, 60.0
# published myCobot 280 joint ranges (deg): used when the robot does not report a usable limit (pymycobot returns -1 on failure)
FALLBACK_LIMITS = [(-168.0, 168.0), (-135.0, 135.0), (-150.0, 150.0), (-145.0, 145.0), (-165.0, 165.0), (-175.0, 175.0)]
LIMIT_MARGIN = 3.0
LAG_SEARCH_MAX_S = 3.0


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
                lo, hi = self.mc.get_joint_min_angle(j), self.mc.get_joint_max_angle(j)
                limits.append((float(lo), float(hi)) if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) else None)
            except Exception:
                limits.append(None)
        return limits

    def info(self):
        out = {"class": self.name}
        try:
            from importlib.metadata import version
            out["pymycobot_version"] = version("pymycobot")
        except Exception:
            out["pymycobot_version"] = "unknown"
        out["python"] = sys.version.split()[0]
        for fn in ("get_system_version", "get_basic_version", "get_robot_version", "get_atom_version"):
            try:
                out[fn] = getattr(self.mc, fn)()
            except Exception:
                pass
        return out


class FakeRobot:
    """Simulated arm for --dry-run. The numbers are invented; it only exercises this script.
    slow=True mimics the problems seen on the real arm: sends that sometimes block, and feedback that freezes after a send."""
    name = "FakeRobot"

    def __init__(self, slow=False):
        self.pos = [0.0, 20.0, -30.0, 10.0, 0.0, 0.0]
        self.target, self.speed = list(self.pos), 40.0
        self.t_last = time.perf_counter()
        self.fb, self.t_fb, self.fb_period, self.frozen_until = list(self.pos), self.t_last, 0.008, 0.0
        self.slow, self.dps_per_speed, self.cmd_latency, self.pending = slow, 1.4, 0.15, []

    def _advance(self, now):
        while self.t_last < now:
            dt = min(0.005, now - self.t_last)
            self.t_last += dt
            for p in [p for p in self.pending if p[0] <= self.t_last]:
                self.target, self.speed = p[1], p[2]
                self.pending.remove(p)
            vmax = self.speed * self.dps_per_speed
            for i in range(6):
                self.pos[i] += max(-vmax, min(vmax, 8.0 * (self.target[i] - self.pos[i]))) * dt

    def get_angles(self):
        time.sleep(0.008)
        now = time.perf_counter()
        self._advance(now)
        if now >= self.frozen_until and now - self.t_fb >= self.fb_period:
            self.fb, self.t_fb = list(self.pos), now
        return list(self.fb)

    def send_angles(self, angles, speed):
        time.sleep(0.45 if self.slow and random.random() < 0.25 else 0.003)
        now = time.perf_counter()
        self._advance(now)
        self.pending.append((now + self.cmd_latency, [float(a) for a in angles], float(speed)))
        if self.slow and random.random() < 0.5:
            self.frozen_until = now + 0.45

    def stop(self):
        self.pending, self.target = [], list(self.pos)

    def set_fresh_mode(self, mode):
        return True

    def joint_limits(self):
        return [(-168, 168), (-140, 140), (-150, 150), (-150, 150), (-155, 160), (-180, 180)]

    def info(self):
        return {"class": "FakeRobot (dry run)", "pymycobot_version": "n/a"}


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


def med(vals):
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 4) if vals else None


def now():
    return time.perf_counter()


def timed_read(robot, t_ref):
    t0 = now()
    a = robot.get_angles()
    t1 = now()
    return t0 - t_ref, t1 - t_ref, a


def read_angles(robot, tries=10):
    for _ in range(tries):
        a = robot.get_angles()
        if a is not None:
            return a
        time.sleep(0.05)
    sys.exit("Could not read joint angles from the robot. Check the port, baud rate, power and firmware.")


def sanitize_limits(raw):
    out = []
    for j, lim in enumerate(raw):
        ok = (lim is not None and len(lim) == 2 and all(isinstance(v, (int, float)) for v in lim)
              and lim[0] < 0.0 < lim[1] and lim[1] - lim[0] >= 2 * (MAX_AMP + LIMIT_MARGIN))
        if lim is not None and not ok:
            print("J%d: robot returned limits %s, ignoring -> using the published 280 default %s" % (j + 1, tuple(lim), FALLBACK_LIMITS[j]))
        out.append(tuple(lim) if ok else None)
    return out


def lim(limits, j):
    return limits[j] if limits[j] else FALLBACK_LIMITS[j]


def clamp_pose(pose, limits):
    return [max(lim(limits, i)[0] + LIMIT_MARGIN, min(lim(limits, i)[1] - LIMIT_MARGIN, a)) for i, a in enumerate(pose)]


def wait_until_at(robot, target, tol=1.5, timeout=12.0):
    t0 = now()
    while now() - t0 < timeout:
        a = robot.get_angles()
        if a is not None and all(abs(x - y) < tol for x, y in zip(a, target)):
            time.sleep(0.3)
            return True
        time.sleep(0.05)
    return False


def go_to(robot, pose, speed=25):
    robot.send_angles(pose, speed)
    return wait_until_at(robot, pose)


def room_sign(pose, j, amp, limits):
    """+1 or -1: the side of joint j with room for an `amp` step (prefers the side with more room), 0 if neither."""
    lo, hi = lim(limits, j)
    up, down = hi - LIMIT_MARGIN - pose[j], pose[j] - (lo + LIMIT_MARGIN)
    if max(up, down) < amp:
        return 0
    return 1 if up >= down else -1


# ------------------------------------------------------------------ analysis of one step trace
def analyze_step(samples, y0, y_target, t_send_end):
    """samples: list of (t_start, t_end, angle) of get_angles() calls after the send. Times are seconds from the start of send_angles().
    A trace is only used for speed/rise numbers when the feedback really updated during the move (we never interpolate across a
    freeze): feedback == 'ok' needs >= 6 value changes inside the move window, no gap between changes above 80 ms, and no single
    update covering more than 30% of the move (a freeze that ends mid-motion shows up as one big jump)."""
    if len(samples) < 8:
        return {"feedback": "too_few_samples", "ok": False}
    ts = [s[1] for s in samples]
    ys = [s[2] for s in samples]
    tail = [y for t, y in zip(ts, ys) if t >= ts[-1] - 0.4] or ys[-3:]
    final = statistics.median(tail)
    delta, commanded = final - y0, y_target - y0
    if abs(commanded) < 0.5:
        return {"feedback": "n/a", "ok": False, "reason": "nothing_to_move"}
    if abs(delta) < max(0.3 * abs(commanded), 0.5):
        return {"feedback": "n/a", "ok": False, "reason": "little_or_no_motion", "moved_deg": round(delta, 2)}
    res = {"ok": True, "moved_deg": round(delta, 2), "commanded_deg": round(commanded, 2), "send_return_s": round(t_send_end, 4),
           "max_read_call_s": round(max(s[1] - s[0] for s in samples), 4)}
    sgn = 1.0 if delta > 0 else -1.0
    prog = [sgn * (y - y0) / abs(delta) for y in ys]                  # 0 -> 1 progress
    i_start = next((i for i, y in enumerate(ys) if abs(y - y0) > 0.3), None)
    i_end = next((i for i, p in enumerate(prog) if p >= 0.95), None)
    if i_start is None or i_end is None or i_end < i_start:
        res.update(feedback="no_motion_seen", ok=False)
        return res
    res["first_motion_s"] = round(ts[i_start], 4)                      # from the start of send_angles()
    res["first_motion_after_send_return_s"] = round(ts[i_start] - t_send_end, 4)
    res["time_to_95pct_s"] = round(ts[i_end], 4)
    changes = [ts[i] for i in range(max(i_start, 1), i_end + 1) if ys[i] != ys[i - 1]]
    gaps = [b - a for a, b in zip(changes, changes[1:])]
    res["feedback_changes_in_move"] = len(changes)
    res["max_feedback_gap_s"] = round(max(gaps), 4) if gaps else None
    # the feedback must also have shown the START of the move: a freeze that ends mid-motion gives one big jump in progress
    max_jump = max((prog[i] - prog[i - 1] for i in range(max(i_start, 1), i_end + 1)), default=1.0)
    res["max_jump_fraction"] = round(max_jump, 3)
    clean = len(changes) >= 6 and gaps and max(gaps) <= 0.08 and max_jump <= 0.30
    res["feedback"] = "ok" if clean else "frozen_or_sparse"
    if not clean:
        return res

    def cross(frac):
        for i in range(i_start, len(ts)):
            if prog[i] >= frac:
                if i == 0 or prog[i] == prog[i - 1]:
                    return ts[i]
                return ts[i - 1] + (ts[i] - ts[i - 1]) * (frac - prog[i - 1]) / (prog[i] - prog[i - 1])
        return None

    t10, t25, t75, t90 = cross(0.10), cross(0.25), cross(0.75), cross(0.90)
    if t10 is not None and t90 is not None and t90 > t10:
        res["rise_10_90_s"] = round(t90 - t10, 4)
    if t25 is not None and t75 is not None and t75 > t25:
        res["speed_25_75_dps"] = round(0.5 * abs(delta) / (t75 - t25), 2)    # near the plateau on a big step
    peak, j = 0.0, 0
    for i in range(len(ts)):
        while j < len(ts) and ts[j] - ts[i] < 0.1:
            j += 1
        if j >= len(ts):
            break
        peak = max(peak, abs(ys[j] - ys[i]) / (ts[j] - ts[i]))
    res["peak_speed_100ms_dps"] = round(peak, 2)
    over = (max(ys) - final) if delta > 0 else (final - min(ys))
    res["overshoot_pct"] = round(max(0.0, over) / abs(delta) * 100.0, 2)
    return res


# ------------------------------------------------------------------ tests
def test_read(robot, n):
    durs, fails, t_start = [], 0, now()
    for _ in range(n):
        t0 = now()
        a = robot.get_angles()
        (durs if a is not None else []).append((now() - t0) * 1000.0)
        fails += a is None
    return {"calls": n, "failures": fails, "call_ms": summarize(durs), "max_read_rate_hz": round(n / (now() - t_start), 2)}


def test_send_only(robot, pose, j, limits, n, speed, rate_hz, dither):
    """n sends at a fixed rate with NO reads. dither=True alternates the target by +-1 deg on joint j so the arm really moves."""
    period, durs, late = 1.0 / rate_hz, [], 0
    base = list(pose)
    t_ref = now()
    next_t = t_ref
    for k in range(n):
        tgt = list(base)
        if dither:
            tgt[j] = base[j] + (1.0 if k % 2 == 0 else -1.0)
        t0 = now()
        robot.send_angles(tgt, speed)
        durs.append((now() - t0) * 1000.0)
        next_t += period
        w = next_t - now()
        if w > 0:
            time.sleep(w)
        else:
            late += 1
            next_t = now()
    total = now() - t_ref
    s = summarize(durs)
    s2 = {"requested_rate_hz": rate_hz, "dither": dither, "n": n, "achieved_rate_hz": round(n / total, 2), "send_ms": s,
          "frac_sends_over_100ms": round(sum(1 for d in durs if d > 100) / n, 3), "late_fraction": round(late / n, 3)}
    s2["sustainable"] = bool(s["p95"] <= 0.8 * period * 1000.0 and s2["achieved_rate_hz"] >= 0.9 * rate_hz)
    return s2


def test_send_then_read(robot, pose, n, speed, delays):
    """send, wait `delay`, read. If the read time drops as the delay grows, the read is waiting for the arm to finish the command."""
    out = []
    for d in delays:
        sends, reads = [], []
        for _ in range(n):
            t0 = now()
            robot.send_angles(pose, speed)
            t1 = now()
            time.sleep(d)
            t2 = now()
            robot.get_angles()
            t3 = now()
            sends.append((t1 - t0) * 1000.0)
            reads.append((t3 - t2) * 1000.0)
        out.append({"wait_s": d, "send_ms": summarize(sends), "read_after_send_ms": summarize(reads)})
    return out


def run_step(robot, base_pose, limits, j, speed, amp, duration, rows, pose_label):
    results = []
    if not go_to(robot, base_pose):
        print("   warning: could not confirm the arm reached the base pose")
    sign = room_sign(base_pose, j, amp, limits)
    if sign == 0:
        return [{"ok": False, "feedback": "n/a", "reason": "no_room", "joint": j + 1, "speed": speed, "amp": amp, "pose": pose_label,
                 "phase": "out"}]
    out_pose = list(base_pose)
    out_pose[j] = base_pose[j] + sign * amp
    for phase, tgt in (("out", out_pose), ("back", list(base_pose))):
        y0 = read_angles(robot)[j]
        t_ref = now()
        robot.send_angles(tgt, speed)
        t_send_end = now() - t_ref
        samples = []
        while now() - t_ref < duration:
            t0, t1, a = timed_read(robot, t_ref)
            if a is not None:
                samples.append((t0, t1, a[j]))
        res = analyze_step(samples, y0, tgt[j], t_send_end)
        res.update({"joint": j + 1, "speed": speed, "amp": amp, "pose": pose_label, "phase": phase, "direction": sign if phase == "out" else -sign})
        results.append(res)
        for t0, t1, y in samples:
            rows.append([pose_label, j + 1, amp, speed, phase, round(t_send_end, 5), round(t0, 5), round(t1, 5), round(y, 4)])
    return results


def stream_analysis(meas, home, joints, amp, f_sig, phases, warmup):
    per_joint = []
    for k, j in enumerate(joints):
        pts = [(t, a[j]) for t, a in meas if t >= warmup]
        if len(pts) < 10:
            continue

        def cmd(t, k=k, j=j):
            return home[j] + amp * math.sin(2 * math.pi * f_sig * t + phases[k])

        best, lag = None, 0.0
        while lag <= LAG_SEARCH_MAX_S:
            err = math.sqrt(statistics.fmean((y - cmd(t - lag)) ** 2 for t, y in pts))
            if best is None or err < best[1]:
                best = (lag, err)
            lag += 0.005
        lag_s, rmse = best
        cm, ym = [cmd(t - lag_s) for t, _ in pts], [y for _, y in pts]
        mc_, my = statistics.fmean(cm), statistics.fmean(ym)
        var = sum((c - mc_) ** 2 for c in cm)
        gain = sum((c - mc_) * (y - my) for c, y in zip(cm, ym)) / var if var > 0 else None
        per_joint.append({"joint": j + 1, "lag_s": round(lag_s, 4), "lag_at_search_limit": lag_s >= LAG_SEARCH_MAX_S - 0.02,
                          "gain": round(gain, 3) if gain else None, "rmse_at_best_lag_deg": round(rmse, 3),
                          "rmse_no_lag_deg": round(math.sqrt(statistics.fmean((y - cmd(t)) ** 2 for t, y in pts)), 3)})
    return per_joint


def run_stream(robot, home, limits, joints, rate, mode, read_every, amp, f_sig, duration, speed, rows):
    ok_mode = robot.set_fresh_mode(mode)
    go_to(robot, home)
    period = 1.0 / rate
    phases = [2 * math.pi * k / max(1, len(joints)) for k in range(len(joints))]
    meas, send_ms, read_ms, late, k = [], [], [], 0, 0
    t_ref = now()
    next_t = t_ref
    while True:
        el = now() - t_ref
        if el >= duration:
            break
        cmd = list(home)
        for i, j in enumerate(joints):
            cmd[j] = home[j] + amp * math.sin(2 * math.pi * f_sig * el + phases[i])
        cmd = clamp_pose(cmd, limits)
        t0 = now()
        robot.send_angles(cmd, speed)
        send_ms.append((now() - t0) * 1000.0)
        if k % read_every == 0:
            t0r, t1r, a = timed_read(robot, t_ref)
            read_ms.append((t1r - t0r) * 1000.0)
            if a is not None:
                meas.append((t1r, a))
                rows.append([rate, mode, read_every, round(el, 5), round(t0r, 5), round(t1r, 5)] + [round(x, 3) for x in cmd] + [round(x, 3) for x in a])
        k += 1
        next_t += period
        w = next_t - now()
        if w > 0:
            time.sleep(w)
        else:
            late += 1
            next_t = now()                                  # a real control loop skips; it does not burst to catch up
    per_joint = stream_analysis(meas, home, joints, amp, f_sig, phases, warmup=1.0 / f_sig)
    stale = sum(1 for (_, a), (_, b) in zip(meas, meas[1:]) if a == b) / max(1, len(meas) - 1)
    res = {"requested_rate_hz": rate, "fresh_mode": mode, "fresh_mode_accepted": ok_mode, "read_every_n_ticks": read_every,
           "ticks": k, "achieved_rate_hz": round(k / duration, 2), "late_fraction": round(late / max(1, k), 3),
           "send_ms": summarize(send_ms), "read_ms": summarize(read_ms), "stale_read_fraction": round(stale, 3),
           "reads": len(meas), "per_joint": per_joint, "ok": bool(per_joint)}
    if per_joint:
        res["median_lag_s"] = med([p["lag_s"] for p in per_joint])
        res["lag_hit_search_limit"] = any(p["lag_at_search_limit"] for p in per_joint)
        res["median_gain"] = med([p["gain"] for p in per_joint])
        res["median_rmse_deg"] = med([p["rmse_at_best_lag_deg"] for p in per_joint])
        res["rmse_over_amp"] = round(res["median_rmse_deg"] / amp, 3)
    return res


# ------------------------------------------------------------------ summary
def make_suggestions(summary):
    out = {"note": "Starting points only. Read summary.json (especially 'warnings') before using any of this in the simulator."}
    info = summary.get("info", {})
    lims = info.get("joint_limits")
    out["joint_limits_deg"] = lims
    out["joint_limits_source"] = info.get("joint_limits_source")
    steps = [r for r in summary.get("step", []) if r.get("pose") == "A"]
    valid = [r for r in steps if r.get("feedback") == "ok"]
    out["step_traces"] = {"total": len(steps), "clean_feedback": len(valid),
                          "frozen_or_sparse": sum(1 for r in steps if r.get("feedback") == "frozen_or_sparse")}
    out["command_delay_s"] = {"from_send_start_median": med([r.get("first_motion_s") for r in valid]),
                              "from_send_return_median": med([r.get("first_motion_after_send_return_s") for r in valid])}
    by = {}
    for r in valid:
        by.setdefault("amp%g_speed%d" % (r["amp"], r["speed"]), []).append(r)
    out["by_amp_and_speed"] = {k: {"n": len(v), "rise_10_90_s": med([x.get("rise_10_90_s") for x in v]),
                                   "speed_25_75_dps": med([x.get("speed_25_75_dps") for x in v]),
                                   "peak_speed_dps": med([x.get("peak_speed_100ms_dps") for x in v]),
                                   "overshoot_pct": med([x.get("overshoot_pct") for x in v])} for k, v in sorted(by.items())}
    sends = summary.get("send_only", [])
    out["sustainable_send_rates_hz"] = sorted({s["requested_rate_hz"] for s in sends if s.get("sustainable")})
    if "send_then_read" in summary:
        out["read_after_send_median_ms_by_wait_s"] = {str(x["wait_s"]): x["read_after_send_ms"].get("median") for x in summary["send_then_read"]}
    return out


def build_warnings(summary):
    w = []
    steps = [r for r in summary.get("step", []) if r.get("feedback") in ("ok", "frozen_or_sparse")]
    if steps:
        bad = sum(1 for r in steps if r["feedback"] != "ok")
        if bad > 0.25 * len(steps):
            w.append("%d of %d step traces had frozen/sparse angle feedback and were NOT used for speed or rise time." % (bad, len(steps)))
    if summary.get("send_only") and not any(s.get("sustainable") for s in summary["send_only"]):
        w.append("No tested send rate was sustainable with send-only streaming: the arm cannot be driven at 10 Hz or more with this setup.")
    for r in summary.get("stream", []):
        if r.get("lag_hit_search_limit"):
            w.append("stream %g Hz mode %s read_every %s: lag reached the %.0f s search limit, the real lag is larger." % (
                r["requested_rate_hz"], r["fresh_mode"], r["read_every_n_ticks"], LAG_SEARCH_MAX_S))
    lims, fb = summary.get("info", {}).get("joint_limits"), FALLBACK_LIMITS
    if lims and all(lims):
        diff = ["J%d %s vs published %s" % (i + 1, tuple(l), fb[i]) for i, l in enumerate(lims) if tuple(l) != fb[i]]
        if diff:
            w.append("Robot-reported joint limits differ from the published 280 table: " + "; ".join(diff))
    return w


def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ------------------------------------------------------------------ selftest of the analysis code (no arm)
def selftest():
    def trace(kind):
        s, t = [], 0.0
        while t < 3.0:
            t += 0.008
            if kind == "clean":                          # first-order rise to 30 deg, starts moving at 0.15 s
                y = 30.0 * (1 - math.exp(-(t - 0.15) / 0.12)) if t > 0.15 else 0.0
            elif kind == "frozen":                       # feedback frozen until 0.5 s, then jumps to the end value
                y = 0.0 if t < 0.5 else 30.0
            else:                                        # frozen until 0.5 s, then follows the real (already running) motion
                y = 0.0 if t < 0.5 else 30.0 * (1 - math.exp(-(t - 0.15) / 0.12))
            s.append((t - 0.008, t, round(y, 2)))
        return s
    a = analyze_step(trace("clean"), 0.0, 30.0, 0.004)
    assert a["feedback"] == "ok" and 0.1 < a["first_motion_s"] < 0.25 and 0.2 < a["rise_10_90_s"] < 0.35, a
    b = analyze_step(trace("frozen"), 0.0, 30.0, 0.004)
    assert b["feedback"] == "frozen_or_sparse" and "rise_10_90_s" not in b, b
    c = analyze_step(trace("late"), 0.0, 30.0, 0.004)
    assert c["feedback"] == "frozen_or_sparse" and "rise_10_90_s" not in c, c
    assert room_sign([0, 0, -92, 0, 0, 0], 2, 30, [None] * 6) == 1 and room_sign([0, 0, 140, 0, 0, 0], 2, 30, [None] * 6) == -1
    print("selftest OK: clean trace accepted (rise %.3f s); frozen and late-unfreeze traces rejected (no speed computed)" % a["rise_10_90_s"])


# ------------------------------------------------------------------ main
def parse_pose(txt):
    v = [float(x) for x in txt.split(",")]
    if len(v) != 6:
        sys.exit("--pose-b needs 6 comma-separated angles in degrees")
    return v


def main():
    ap = argparse.ArgumentParser(description="Measure a real myCobot 280 so a simulator can be tuned to it (v2).")
    ap.add_argument("--port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dry-run", action="store_true", help="Use a simulated arm (no robot needed)")
    ap.add_argument("--fake-slow", action="store_true", help="With --dry-run: mimic blocking sends and frozen feedback")
    ap.add_argument("--selftest", action="store_true", help="Check the analysis code and exit")
    ap.add_argument("--quick", action="store_true", help="Shorter run (for testing the script)")
    ap.add_argument("--yes", action="store_true", help="Skip the safety confirmations")
    ap.add_argument("--amps", type=float, nargs="+", default=[10.0, 30.0], help="Step amplitudes in degrees (default 10 30)")
    ap.add_argument("--allow-large", action="store_true", help="Allow amplitudes above 25 deg (up to 60)")
    ap.add_argument("--speeds", type=int, nargs="+", default=[20, 40, 60], help="Speed settings (1-80) for the step tests")
    ap.add_argument("--joints", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--pose-b", type=parse_pose, default=None, help="second pose in degrees, e.g. 0,30,-30,0,0,0 (steps repeated there)")
    ap.add_argument("--joints-b", type=int, nargs="+", default=[2, 3], help="joints tested in pose B (default 2 3: the gravity-loaded ones)")
    ap.add_argument("--zero-pose", action="store_true", help="also send all joints to 0 and record what the arm reports there")
    ap.add_argument("--rates", type=float, nargs="+", default=[10, 25, 50], help="streaming / send-only rates (Hz)")
    ap.add_argument("--fresh-modes", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--read-every", type=int, nargs="+", default=[1, 5], help="stream test: read once every N ticks")
    ap.add_argument("--stream-speed", type=int, default=60)
    ap.add_argument("--stream-amp", type=float, default=10.0)
    ap.add_argument("--step-duration", type=float, default=3.0)
    ap.add_argument("--stream-duration", type=float, default=8.0)
    ap.add_argument("--out", default="calibration_out")
    ap.add_argument("--skip", nargs="*", default=[], choices=["timing", "step", "stream", "zero"])
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    amps = sorted({abs(a) for a in args.amps})
    cap = MAX_AMP_LARGE if args.allow_large else MAX_AMP
    if any(a > cap for a in amps):
        print("Amplitudes above %.0f deg are capped (use --allow-large for up to %.0f)." % (cap, MAX_AMP_LARGE))
        amps = sorted({min(a, cap) for a in amps})
    speeds = [max(1, min(80, s)) for s in args.speeds]
    joints = [j - 1 for j in args.joints if 1 <= j <= 6]
    joints_b = [j - 1 for j in args.joints_b if 1 <= j <= 6]
    rates, fresh_modes, read_everys = args.rates, args.fresh_modes, args.read_every
    step_dur, stream_dur, n_read, n_send, n_str = args.step_duration, args.stream_duration, 200, 60, 12
    if args.quick:
        joints, joints_b, speeds, amps = joints[:1], joints_b[:1], speeds[1:2] or speeds[:1], amps[:1]
        rates, fresh_modes, read_everys = rates[:1], fresh_modes[:1], read_everys[:1]
        step_dur, stream_dur, n_read, n_send, n_str = 1.5, 3.0, 40, 20, 5

    if args.dry_run:
        robot = FakeRobot(slow=args.fake_slow)
    else:
        port = args.port
        if not port:
            try:
                from serial.tools import list_ports
                ports = [p.device for p in list_ports.comports()]
            except Exception:
                ports = []
            if len(ports) != 1:
                sys.exit("Pass --port explicitly. Serial ports found: %s" % (ports or "none"))
            port = ports[0]
        robot = RealRobot(port, args.baud)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, stamp)
    os.makedirs(out_dir, exist_ok=True)
    limits = sanitize_limits(robot.joint_limits())
    home = read_angles(robot)
    info = {"timestamp": stamp, "version": "v2", "dry_run": args.dry_run, "robot": robot.info(), "home_angles_deg": home,
            "joint_limits": [list(l) if l else None for l in limits],
            "joint_limits_source": "robot" if all(limits) else "partly missing, fallback used",
            "settings": {"amps_deg": amps, "speeds": speeds, "rates": rates, "fresh_modes": fresh_modes, "read_every": read_everys,
                         "stream_speed": args.stream_speed, "joints": [j + 1 for j in joints], "pose_b": args.pose_b}}
    biggest = max(amps)
    safe = [j for j in joints if room_sign(home, j, biggest, limits) != 0]
    for j in joints:
        if j not in safe:
            print("Skipping joint %d: no side has room for a %.0f deg step from this pose." % (j + 1, biggest))
    joints = safe
    if not joints:
        sys.exit("No joint can move %.0f deg from this pose. Move the arm to a more central pose, or use a smaller --amps." % biggest)
    pose_b = clamp_pose(args.pose_b, limits) if args.pose_b else None

    print("\nStarting pose A (deg):", [round(a, 1) for a in home])
    print("Each tested joint steps up to %s deg, one joint at a time, toward the side with more room. Speeds: %s." % (amps, speeds))
    if pose_b:
        print("Pose B (deg):", [round(a, 1) for a in pose_b], "- the arm will be sent there slowly; joints tested there:", [j + 1 for j in joints_b])
    if args.zero_pose and "zero" not in args.skip:
        print("--zero-pose: the arm will be sent to all zeros (upright). Needs free space above the base.")
    print("Make sure nothing (table, board, person) is within reach. Ctrl+C stops the arm.")
    if not args.yes:
        input("Press Enter to start... ")

    summary, interrupted = {"info": info}, False
    try:
        if "timing" not in args.skip:
            print("\n[1/4] Timing of single calls ...")
            summary["read"] = test_read(robot, n_read)
            print("      read: median %.1f ms, p95 %.1f ms, failures %d" % (summary["read"]["call_ms"].get("median", float("nan")),
                  summary["read"]["call_ms"].get("p95", float("nan")), summary["read"]["failures"]))
            dj = joints[0]
            summary["send_only"] = []
            for mode in fresh_modes:
                robot.set_fresh_mode(mode)
                for rate in rates:
                    for dither in (False, True):
                        r = test_send_only(robot, home, dj, limits, n_send, speeds[0], rate, dither)
                        r["fresh_mode"] = mode
                        summary["send_only"].append(r)
                        print("      send-only mode %d @ %g Hz%s: achieved %.1f Hz, send median %.1f ms, p95 %.1f ms, %.0f%% over 100 ms -> %s" % (
                            mode, rate, " (dither)" if dither else "", r["achieved_rate_hz"], r["send_ms"]["median"], r["send_ms"]["p95"],
                            100 * r["frac_sends_over_100ms"], "OK" if r["sustainable"] else "NOT sustainable"))
            robot.set_fresh_mode(fresh_modes[0])
            summary["send_then_read"] = test_send_then_read(robot, home, n_str, speeds[0], [0.0, 0.1, 0.3, 0.6, 1.0] if not args.quick else [0.0, 0.3])
            for x in summary["send_then_read"]:
                print("      send, wait %.1f s, read: read takes median %.1f ms (send %.1f ms)" % (
                    x["wait_s"], x["read_after_send_ms"]["median"], x["send_ms"]["median"]))
            go_to(robot, home)
        if "step" not in args.skip:
            print("[2/4] Step response ...")
            rows, results = [], []
            plan = [("A", home, joints, amps)] + ([("B", pose_b, joints_b, [biggest])] if pose_b else [])
            for label, base, js, am in plan:
                if label == "B":
                    print("      moving to pose B ...")
                    if not go_to(robot, base, speed=20):
                        print("      warning: pose B not confirmed")
                for j in js:
                    for a in am:
                        for s in speeds:
                            r = run_step(robot, base, limits, j, s, a, step_dur, rows, label)
                            results += r
                            for x in r:
                                msg = ("rise %.3fs, delay %.3fs, plateau %s deg/s" % (x.get("rise_10_90_s", float("nan")), x.get("first_motion_s", float("nan")),
                                       x.get("speed_25_75_dps")) if x.get("feedback") == "ok" else x.get("feedback") or x.get("reason"))
                                print("      pose %s J%d amp %g speed %d %s: %s" % (label, j + 1, a, s, x.get("phase"), msg))
                if label == "B":
                    go_to(robot, home, speed=20)
            summary["step"] = results
            save_csv(os.path.join(out_dir, "step_response.csv"),
                     ["pose", "joint", "amp_deg", "speed", "phase", "send_return_s", "read_start_s", "read_end_s", "angle_deg"], rows)
        if "stream" not in args.skip:
            print("[3/4] Streamed tracking (%d rates x %d modes x %d read patterns) ..." % (len(rates), len(fresh_modes), len(read_everys)))
            rows, results = [], []
            for mode in fresh_modes:
                for rate in rates:
                    for re_ in read_everys:
                        r = run_stream(robot, home, limits, joints[:3], rate, mode, re_, args.stream_amp, 0.4, stream_dur, args.stream_speed, rows)
                        results.append(r)
                        print("      mode %d @ %g Hz, read every %d: achieved %.1f Hz, send p95 %.0f ms, lag %s s%s, rmse/amp %s" % (
                            mode, rate, re_, r["achieved_rate_hz"], r["send_ms"].get("p95", float("nan")), r.get("median_lag_s"),
                            " (>= limit)" if r.get("lag_hit_search_limit") else "", r.get("rmse_over_amp")))
            summary["stream"] = results
            header = ["rate_hz", "fresh_mode", "read_every", "t_send_s", "read_start_s", "read_end_s"] + \
                     ["cmd_j%d" % i for i in range(1, 7)] + ["meas_j%d" % i for i in range(1, 7)]
            save_csv(os.path.join(out_dir, "stream_tracking.csv"), header, rows)
        if args.zero_pose and "zero" not in args.skip:
            print("[4/4] Zero pose ...")
            if not args.yes:
                input("The arm will go UPRIGHT (all joints 0). Is there free space above it? Press Enter to continue... ")
            ok = go_to(robot, [0.0] * 6, speed=20)
            time.sleep(1.0)
            summary["zero_pose"] = {"reached": ok, "reported_angles_deg": read_angles(robot),
                                    "how_to_use": "photograph the arm now and compare it with the simulator's q=0 pose"}
            print("      reported angles at zero command:", [round(a, 2) for a in summary["zero_pose"]["reported_angles_deg"]])
            go_to(robot, home, speed=20)
    except KeyboardInterrupt:
        interrupted = True
        robot.stop()
        print("\nInterrupted: arm stopped. Saving what was measured so far.")
    finally:
        if not interrupted:
            try:
                go_to(robot, home)
            except Exception:
                pass
        summary["interrupted"] = interrupted
        summary["warnings"] = build_warnings(summary)
        save_json(os.path.join(out_dir, "summary.json"), summary)
        save_json(os.path.join(out_dir, "suggested_sim_params.json"), make_suggestions(summary))
    for w in summary["warnings"]:
        print("WARNING:", w)
    print("\nDone. Results saved in: %s\nBring back the whole folder (summary.json, suggested_sim_params.json and the CSV logs)." % out_dir)


if __name__ == "__main__":
    main()