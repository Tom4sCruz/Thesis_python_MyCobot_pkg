"""
Move the end effector along a chosen y-z path (LINEAR or CURVED) at fixed x,
running it back and forth once per JERK PRESET, to compare how different
jerk / twitch settings feel.

This is the jerk analogue of Accel_Test/test_accel.py: there the lever was the
per-segment `durations` list; here it is the three jerk knobs, swept over a
list of presets so you can feel them one after another on the same path.

arm.jerk / arm.random_twitch / arm.twitch_intensity apply to every motion path
now (send_coords, send_path, move_joints; normal AND single-joint execution):

    arm.jerk             roughness dial, 0 = smooth (tremor + uneven pace)
    arm.random_twitch    probability [0, 1] of a flinch per streamed setpoint
    arm.twitch_intensity peak flinch amplitude, in degrees
    arm.jerk_seed        None = fresh randomness each run; set an int to make a
                         run exactly reproducible

The move BACK to the start point before each preset is always run clean (jerk
off); only the path itself is perturbed. Optionally dumps each run's streamed
joint trace to CSV (arm.last_execution.to_csv) for offline analysis.
"""
import os
import time

import numpy as np

from armik import Arm, config

arm = Arm("/dev/ttyTHS1", 1000000)
HOME = [0, 0, -90, 0, 0, 0]

LINEAR = [(-20, 5), (-8, 15), (8, 12), (20, 5)]

STARTING_Y = -20

# label, jerk, random_twitch, twitch_intensity -- edit freely
PRESETS = [
    ("smooth",          0.0, 0.00,  0.0),
    ("mild-tremor",     2.0, 0.00,  0.0),
    ("strong-tremor",   5.0, 0.00,  0.0),
    ("twitchy",         0.0, 0.15, 12.0),
    ("tremor+twitch",   6.0, 0.20, 12.0),
]


def get_CURVED(min_y=STARTING_Y, max_y=-STARTING_Y):
    def z(y):
        return round(8 * np.sin(np.pi / 20 * y + np.pi) + 8, 2)

    y = min_y
    res = []
    while y <= max_y:
        res.append((y, z(y)))
        y += 1
    return res


def exec_path(x, points, cycles, **send_path_kwargs):
    ys = [p[0] for p in points]
    zs = [p[1] for p in points]
    for i in range(cycles):
        target_y = ys[::-1] if i % 2 == 1 else ys
        target_z = zs[::-1] if i % 2 == 1 else zs
        r = arm.send_path(x=x, y=target_y, z=target_z, **send_path_kwargs)
        if not r:
            print(f"  cycle {i}: path failed -- {arm.last_error}")


def choose_path():
    choice = input("Which path? [L]INEAR / [C]URVED: ").strip().lower()
    if not choice.startswith("l"):
        return "CURVED", get_CURVED()
    return "LINEAR", LINEAR


def prompt_float(label, current):
    raw = input(f"{label} [{current}]: ").strip()
    return float(raw) if raw else current


def clear_jerk():
    arm.jerk = 0.0
    arm.random_twitch = 0.0
    arm.twitch_intensity = 0.0


def run_preset(idx, preset, cur_x, points, cycles, speed, single_joint, seed, csv_dir):
    label, jerk, random_twitch, twitch_intensity = preset
    print(f"\n=== [{idx}] {label} ===")

    # approach the start point clean -- no jitter on the repositioning move
    clear_jerk()
    arm.set_single_joint(0)
    print(f"  moving to start ({STARTING_Y}, 8) ...")
    arm.send_coords(x=cur_x, y=STARTING_Y, z=8, speed=6)
    time.sleep(0.2)

    # arm the jerk for the path itself
    arm.jerk = jerk
    arm.random_twitch = random_twitch
    arm.twitch_intensity = twitch_intensity
    arm.jerk_seed = seed
    arm.set_single_joint(1 if single_joint else 0)
    print(f"  jerk={jerk}  random_twitch={random_twitch}  "
          f"twitch_intensity={twitch_intensity}  seed={seed}  "
          f"single_joint={single_joint}")

    if input("  [enter] to run, 's' to skip: ").strip().lower() == "s":
        return
    exec_path(cur_x, points, cycles, speed=speed)

    ex = arm.last_execution
    if ex is None:
        return
    print(f"  {ex.setpoints} setpoints, {ex.late_deadlines} late deadlines, "
          f"{ex.duration_s:.2f}s")
    if csv_dir:
        path = os.path.join(csv_dir, f"jerk_{idx:02d}_{label}.csv")
        try:
            ex.to_csv(path)
            print(f"  wrote {path}")
        except OSError as exc:
            print(f"  could not write csv: {exc}")


def main():
    print("moving to HOME...")
    time.sleep(0.3)
    clear_jerk()
    arm.move_joints(HOME, duration=2.0)

    current = arm.get_coords()
    cur_x = current[0]

    name, points = choose_path()
    print(f"This is the list of points of {name} ({len(points)} points):")
    print(points)
    if input("[?] Continue? (Y/n): ") not in ["y", "Y", ""]:
        print("Terminating script...")
        return

    single_joint = input("single-joint mode? [y/N]: ").strip().lower().startswith("y")
    cycles = int(input("cycles per preset [default=2]: ") or 2)
    speed = prompt_float("speed (cm/s)", 5.0)

    seed_raw = input("jerk_seed (blank = fresh each run): ").strip()
    seed = int(seed_raw) if seed_raw else None

    csv_dir = input("csv output dir (blank = no csv): ").strip() or None
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    print("\npresets:")
    for i, (label, j, rt, ti) in enumerate(PRESETS):
        print(f"  [{i}] {label:16s} jerk={j}  random_twitch={rt}  twitch_intensity={ti}")
    if input("[?] Continue? (Y/n): ") not in ["y", "Y", ""]:
        return

    for idx, preset in enumerate(PRESETS):
        run_preset(idx, preset, cur_x, points, cycles, speed, single_joint, seed, csv_dir)

    clear_jerk()
    arm.set_single_joint(0)
    print("\nmoving to HOME...")
    time.sleep(0.1)
    arm.move_joints(HOME, duration=2.0)


if __name__ == "__main__":
    main()
