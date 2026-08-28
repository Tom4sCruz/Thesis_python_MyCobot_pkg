"""
Move the end effector along a chosen y-z path (LINEAR or CURVED) at fixed x,
for a few back-and-forth cycles -- to compare how different speed/duration
settings feel. There's no dedicated acceleration control yet (send_path()
only has speed= and durations=), so this exercises those two knobs; per-
segment durations are the more direct lever for feeling acceleration between
points, since a flat speed keeps velocity roughly uniform across segments by
design.
"""
import time
import numpy as np
from armik import Arm, config
import math

arm = Arm("/dev/ttyTHS1", 1000000)
HOME = [0, 0, -90, 0, 0, 0]

LINEAR = [(-20, 0), (-20, 10), (20, 10), (20, 0)]


def get_CURVED(num=30, min_y=-20, max_y=20, cycles=1):
    """
    z(y) = 8*sin(2*pi*cycles*(y-min_y)/(max_y-min_y) + phase) + 8

    cycles=1 gives one full gentle wave across [min_y, max_y] -- almost
    certainly what "like a sine wave" means visually. Your literal formula
    (8*sin(y+20)+8, ~6.4 oscillations over this same range) is one line
    below if you actually want that instead.
    """
    span = max_y - min_y

    def z(y):
        # one full wave across the sweep, by default
        #val = 8 * np.sin(2 * np.pi * cycles * (y - min_y) / span) + 8
        val = 8 * np.sin(np.pi / 20 * y + np.pi) +8 
        # to use your exact literal formula instead, comment the line above
        # and uncomment this one:
        # val = 8 * np.sin(y + 20) + 8
        return round(float(val), 2)

    step = span / num
    y = min_y
    res = []
    while y <= max_y:
        res.append((y, z(y)))
        y += 1
    return res


def exec_path(x, points, cycles, durations):
    ys = [p[0] for p in points]
    zs = [p[1] for p in points]
    for i in range(cycles):
        target_y = ys[::-1] if i % 2 == 1 else ys
        target_z = zs[::-1] if i % 2 == 1 else zs 
        r = arm.send_path(x=x, y=target_y, z=target_z, durations=durations)
        #r = arm.send_path(x=x, y=target_y, z=target_z, speed=10)
        if not r:
            print(f"  cycle {i}: path failed -- {arm.last_error}")


def choose_path():
    choice = input("Which path? [L]INEAR / [C]URVED: ").strip().lower()
    if not choice.startswith("l"):
        return "CURVED", get_CURVED()
    return "LINEAR", LINEAR


def peak_slow_durations(points, max_duration, min_duration, low_pct=5, high_pct=95):
    """
    Same as before, but robust to outlier segments: uses the low_pct/high_pct
    percentiles of the segment distances (not the raw min/max) to set the
    mapping range, and clamps every distance into that range first. A single
    unusually long or short segment (e.g. a big jump between two points that
    aren't really part of the smooth curve) then just gets pinned to
    min_duration or max_duration -- it no longer drags the whole scale with
    it and crushes every other segment's duration toward one end.

    max_duration : seconds for the flattest segment(s).
    min_duration : seconds for the steepest segment(s).
    low_pct/high_pct : percentile bounds used instead of raw min/max.
        5/95 (default) ignores the most extreme ~5% at each end. Use 0/100
        to fall back to exact min/max (no outlier protection).
    """
    dists = np.array([
        math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        for i in range(len(points))
    ])
    dlo, dhi = np.percentile(dists, [low_pct, high_pct])

    durations = []
    for d in dists:
        dc = min(max(d, dlo), dhi)               # clamp outliers into range
        if dhi == dlo:
            durations.append(round(max_duration, 3))
            continue
        t = (dc - dlo) / (dhi - dlo)              # 0 at flattest, 1 at steepest
        durations.append(round(max_duration + t * (min_duration - max_duration), 3))
    return durations


def main():
    print("moving to HOME...")
    time.sleep(0.3)
    arm.move_joints(HOME, duration=2.0)

    current = arm.get_coords()
    cur_x = current[0]

    name, points = choose_path()
    print(f"This is the list of points of {name} ({len(points)} points):")
    print(points)
    if input("[?] Continue? (Y/n): ") not in ["y", "Y", ""]:
        print("Terminating script...")
        return

    # per-segment durations -- edit this list directly to feel acceleration:
    # short duration on one segment + long on the next = a sharp speed change
    # right at that point, exactly where the arm goes fast-then-slow (or the
    # reverse). A flat value here (e.g. [2] * len(points)) behaves like the
    # old speed= approach instead -- roughly uniform pace, no deliberate
    # acceleration/deceleration built into the path itself.
    
    #durations = len(points) * [0.25]
    #durations[8:13] = [0.4] * (13-8)
    #durations[28:33] = [0.4] * (33-28)
    
    durations = peak_slow_durations(points, max_duration=0.15, min_duration=0.08)

    print(f"len(points) = {len(points)} | len(durations) = {len(durations)}")

    print("durations =", durations)
    if input("Correct? [Y/n]: ") not in ["Y", "y", ""]:
        return

    cycles = int(input("cycles: ") or 2)

    print("moving to starting point: (-20,8)")
    arm.send_coords(x=cur_x, y=-20, z=8, speed=8)
    time.sleep(0.1)

    print(f"starting {name}...")
    time.sleep(0.1)
    exec_path(cur_x, points, cycles, durations=durations)

    print("moving to HOME...")
    time.sleep(0.5)
    arm.move_joints(HOME, duration=2.0)


if __name__ == "__main__":
    main()
