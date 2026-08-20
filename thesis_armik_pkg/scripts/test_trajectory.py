import time
from armik import Arm, config

arm = Arm("/dev/ttyTHS1", 1000000)
arm.conn.set_fresh_mode(1)
FRESH_MODE = arm.conn.get_fresh_mode()

HOME = [0, 0, -90, 0, 0, 0]


def get_CURVED(num=30):
    def z(y):
        z_val = -1 / 15 * pow(y, 2) + 15
        return round(float(z_val), 2)

    MIN_Y = -15
    MAX_Y = 15
    NUMBER_OF_X = MAX_Y - MIN_Y

    step = NUMBER_OF_X / num
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
    i = 0
    while i < cycles:
        cycle_points = points if (i % 2 == 0) else points[::-1]
        for p in cycle_points:
            arm.send_coords(x=x, y=p[0], z=p[1], speed=speed)

            if FRESH_MODE:
                time.sleep(0.1)

        i += 1


def main():
    print("moving to HOME...")
    time.sleep(0.5)

    arm.move_joints(HOME, duration=2.0)

    current = arm.get_coords()
    cur_x = current[0]

    LINEAR = [(-15, 0), (-15, 15), (15, 15), (15, 0)]
    CURVED = get_CURVED(120)

    print("starting LINEAR path...")
    time.sleep(0.5)

    exec_path(cur_x, LINEAR, 15, cycles=0)

    print("moving to HOME...")
    time.sleep(0.5)

    arm.move_joints(HOME, duration=2.0)

    print("starting CURVED path...")
    time.sleep(0.5)

    exec_path(cur_x, CURVED, 15, cycles=2)

    print("moving to HOME...")
    time.sleep(0.5)

    arm.move_joints(HOME, duration=2.0)


if __name__ == "__main__":
    main()
