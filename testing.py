# Useful code snippets:
import numpy as np

from panda_py import constants
constants.JOINT_LIMITS_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
constants.JOINT_LIMITS_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
import panda_py
from panda_py import libfranka

panda = panda_py.Panda("172.16.0.2")
gripper = libfranka.Gripper("172.16.0.2")

panda.move_to_start()
pose = panda.get_pose()
print(pose)
pose[0 ,3] -= .1 # in meters ;   todo: suggested:  clip xyz to sy 20-30 cm box;  coords to be calculated
q = panda_py.ik(pose)
panda.move_to_joint_position(q)



exit()