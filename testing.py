# Useful code snippets:
import numpy as np
from panda_py import constants
constants.JOINT_LIMITS_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
constants.JOINT_LIMITS_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
import panda_py
from panda_py import libfranka
import time

GRIPPED = False

# grip tester

panda = panda_py.Panda("172.16.0.2")
gripper = libfranka.Gripper("172.16.0.2")

panda.move_to_start()


# prepare to grip (stage 1)
pose = panda.get_pose()

pose[1, 3] += .1
pose[0, 3] += .19

q = panda_py.ik(pose)
panda.move_to_joint_position(q)

# prepare to grip (lower the arm)
pose = panda.get_pose()
pose[2 ,3] += -.42 # in meters ;   todo: suggested:  clip xyz to say 20-30 cm box;  coords to be calculated
q = panda_py.ik(pose)
panda.move_to_joint_position(q)

# actually grip
if GRIPPED:
    gripper.move(0.08, 0.2) # release gripper
    GRIPPED = False
else:
    gripper.grasp(0, 0.2, 50, 0.04, 0.04) # grip
    GRIPPED = True
time.sleep(1)

# raise the arm again first to prevent pushing the block away
pose = panda.get_pose()
pose[2 ,3] += .42 # in meters ;   todo: suggested:  clip xyz to say 20-30 cm box;  coords to be calculated
q = panda_py.ik(pose)
panda.move_to_joint_position(q)

# reset the arm again
panda.move_to_start()

exit()


## initial starting positions 

# [[ 9.99985257e-01  3.15050973e-03 -5.53410256e-04  3.07600574e-01]
# [ 3.14868280e-03 -9.99980064e-01 -3.27166697e-03 -7.20013878e-05]
# [-5.63706642e-04  3.26987622e-03 -9.99994495e-01  4.86228170e-01]
# [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]