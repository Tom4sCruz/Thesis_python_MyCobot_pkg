"""
Move the end effector along a chosen y-z path (LINEAR or CURVED) at fixed x,
to compare how different jerk/twitch settings feel.

arm.jerk / arm.random_twitch / arm.twitch_intensity apply to EVERY motion path
now -- send_coords(), send_path(), move_joints(), in both normal and
single-joint execution. This script runs in single-joint mode (see
arm.set_single_joint(1) below); comment that call out to feel the same knobs on
the coordinated path instead.

    arm.jerk             roughness dial, 0 = smooth (tremor + uneven pace)
    arm.random_twitch    probability [0, 1] of a flinch per streamed setpoint
    arm.twitch_intensity peak flinch amplitude, in degrees
    arm.jerk_seed        None = fresh each run; set an int to reproduce a run
"""
import time
import numpy as np
from armik import Arm, config

arm = Arm("/dev/ttyTHS1", 1000000)
HOME = [0, 0, -90, 0, 0, 0]

LINEAR = [(-20, 5), (-8, 15), (8, 12), (20, 5)]

STARTING_Y = -20

def get_CURVED(num=30, min_y=STARTING_Y, max_y=-STARTING_Y, cycles=1):
    def z(y):
        return round(8 * np.sin(np.pi/20 * y + np.pi) + 8, 2)

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


    ## Set JERK Params #########
    arm.jerk = 3.0
    arm.random_twitch = 0.0
    arm.twitch_intensity = 0.0
    ############################

    print(f"jerk={arm.jerk}  random_twitch={arm.random_twitch}  twitch_intensity={arm.twitch_intensity}")
    if input("Correct? [Y/n]: ") not in ["Y", "y", ""]:
        return

    durations = len(points) * [2.0]

    cycles = int(input("cycles [default=2]: ") or 2)

    print(f"moving to starting point: ({STARTING_Y}, 8)")
    arm.send_coords(x=cur_x, y=STARTING_Y, z=8, speed=6)
    time.sleep(0.2)

    # SINGLE-JOINT mode
    arm.set_single_joint(1)

    print("starting PATH...")
    time.sleep(0.1)
    exec_path(cur_x, points, cycles, speed=5)

    print("moving to HOME...")
    time.sleep(0.1)
    arm.move_joints(HOME, duration=2.0)


if __name__ == "__main__":
    main()
