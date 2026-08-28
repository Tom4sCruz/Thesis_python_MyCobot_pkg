from armik import Arm, config
import time


arm = Arm("/dev/ttyTHS1", 1000000)
HOME = [0,0,-90, 0,0,0]

arm.move_joints(HOME, duration=2.0)

config.SINGLE_JOINT_DELAY_BETWEEN_POINTS = 0.5

print(f"Between joints delay: {config.SINGLE_JOINT_DELAY}s")
print(f"Between points delay: {config.SINGLE_JOINT_DELAY_BETWEEN_POINTS}s")

input("Waiting to start...")

current = arm.get_coords()
cur_x = current[0]

print("current = ", current, "\ncur_x = ", cur_x)
print("single-joint mode: ", arm.set_single_joint(1))

LINEAR = [(-20, 5), (-8, 15), (8, 12), (20, 4)]

LINEAR_y = [point[0] for point in LINEAR]
LINEAR_z = [point[1] for point in LINEAR]

res = arm.send_path(x=cur_x, y=LINEAR_y, z=LINEAR_z, speed=10)
print(f"\nsend_path: {res}")
print("last_error:", arm.last_error)

time.sleep(config.SINGLE_JOINT_DELAY_BETWEEN_POINTS)

print("Moving to HOME...")
time.sleep(0.5)
arm.move_joints(HOME, duration=2.0)
