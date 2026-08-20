"""
Move the end effector along two y-z paths (LINEAR and CURVED) at fixed x,
each run for a few back-and-forth cycles.
"""

import time

from armik import Arm, config

arm = Arm("/dev/ttyTHS1", 1000000)
HOME = [0, 0, -90, 0, 0, 0]


def get_CURVED(num=30):
    def z(y):
        z = -1 / 15 * pow(y, 2) + 15
        return round(float(z), 2)

    MIN_Y = -15
    MAX_Y = 15
    step = 30 / num
    y = MIN_Y

    res = []
    while y <= MAX_Y:
        res += [(y, z(y))]
        y += step

    print("This is the list of points of CURVED:")
    print(res)
    if input("[?] Which to continue? (Y/n): ") not in ["y", "Y", ""]:
        print("Terminating script...")
        quit()
    return res


def exec_path(x, points, speed, cycles=3):
    ys = [p[0] for p in points]
    zs = [p[1] for p in points]
    i = 0
    while i < cycles:
        cur_ys = ys if (i % 2 == 0) else ys[::-1]
        cur_zs = zs if (i % 2 == 0) else zs[::-1]
        r = arm.send_path(x=x, y=cur_ys, z=cur_zs, speed=speed)
        if not r:
            print(f"  cycle {i}: path failed -- {arm.last_error}")
        i += 1


def main():
    print("moving to HOME...")
    time.sleep(0.5)
    arm.move_joints(HOME, duration=2.0)
    current = arm.get_coords()
    cur_x = current[0]

    LINEAR = [(-15, 0), (-15, 15), (15, 15), (15, 0)]
    CURVED = get_CURVED()

    print("starting LINEAR path...")
    time.sleep(0.5)
    exec_path(cur_x, LINEAR, 15, 2)

    print("moving to HOME...")
    time.sleep(0.5)
    arm.move_joints(HOME, duration=2.0)

    print("starting CURVED path...")
    time.sleep(0.5)
    exec_path(cur_x, CURVED, 15, 2)


if __name__ == "__main__":
    main()
