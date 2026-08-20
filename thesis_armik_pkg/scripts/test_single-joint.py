from armik import Arm, config

arm = Arm("/dev/ttyTHS1", 1000000)
HOME = [0,0,-90, 0,0,0]

arm.move_joints(HOME, duration=2.0)

input("Waiting to start...")

current = arm.get_coords()
cur_x = current[0]

print("current = ", current, "\ncur_x = ", cur_x)
print("single-joint mode: ", arm.set_single_joint(1))

LINEAR = [(-20, 0), (-8, 15), (8, 5), (20, 0)]

LINEAR_y = [point[0] for point in LINEAR]
LINEAR_z = [point[1] for point in LINEAR]

res = arm.send_path(x=cur_x, y=LINEAR_y, z=LINEAR_z, speed=4)
print(f"\nsend_path: {res}")
print("last_error:", arm.last_error)
