"""

PANDA_PY  -- install with "pip install panda-python" ?  it may be libfranka0.9.2 instead of the newer 0.13 though, but it seems to work?

https://www.softxjournal.com/article/S2352-7110(23)00228-5/pdf

https://jeanelsner.github.io/panda-py/panda_py.html

https://github.com/ElsevierSoftwareX/SOFTX-D-23-00483



http://172.16.0.2 -> unlock joints + activate FCI

or

-----
import panda_py
desk = panda_py.Desk(hostname, username, password)
desk.unlock()
desk.activate_fci()
-----
"""


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
pose[2 ,3] -= .1 # in meters ;   todo: suggested:  clip xyz to sy 20-30 cm box;  coords to be calculated
q = panda_py.ik(pose)
panda.move_to_joint_position(q)



exit()



## panda.get_state() returns the python version of this object: https://frankaemika.github.io/libfranka/structfranka_1_1RobotState.html


## NOTE:   move_to_pose moves in cartesian space, while move_to_joint_position moves in joint space;  i.e.,  linear interpolation in pose (cartesian) vs linear interpolation in joints space



## CARTESIAN IMPEDANCE CONTROL:

import numpy as np
from panda_py import constants
constants.JOINT_LIMITS_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
constants.JOINT_LIMITS_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
from panda_py import controllers
import panda_py

panda = panda_py.Panda("172.16.0.2")
panda.move_to_start()

ctrl = controllers.CartesianImpedance()
x0 = panda.get_position()
q0 = panda.get_orientation()

runtime = np.pi*4.0
panda.start_controller(ctrl)
with panda.create_context(frequency=1e3, max_runtime=runtime) as ctx:
    while ctx.ok():
        x_d = x0.copy()
        x_d[1] += 0.1*np.sin(ctrl.get_time())
        ctrl.set_control(x_d, q0)




## ENABLE LOGGING before a motion; logging is stopped when motion ends

panda.move_to_start()
panda.enable_logging(2*1000) # 2s

pose = panda.get_pose()
pose[1 ,3] -= .2
q = panda_py.ik(pose)

#panda.move_to_joint_position(q)
panda.move_to_pose(pose, speed_factor=0.1)

panda.disable_logging()
log = panda.get_log()






#### torque-0 fully compliant mode (to move robot by hand)
import numpy as np
from panda_py import constants
constants.JOINT_LIMITS_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
constants.JOINT_LIMITS_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
from panda_py import controllers
import panda_py
import time

panda = panda_py.Panda("172.16.0.2")

panda.move_to_start()

ctrl = controllers.AppliedTorque()
panda.start_controller(ctrl)


####### OR ALTERNATIVELY (probably a better option)
panda.teaching_mode(True) # False to disable



## EXAMPLE:

LEN = 10
input(
  f'Next, teach a trajectory for {LEN} seconds. Press enter to begin.')
panda.teaching_mode(True)
panda.enable_logging(LEN * 1000)
time.sleep(LEN)
panda.teaching_mode(False)

q = panda.get_log()['q']
dq = panda.get_log()['dq']

# + playback trajectory
input('Press enter to replay trajectory')
panda.move_to_joint_position(q[0])
i = 0
ctrl = controllers.JointPosition()
panda.start_controller(ctrl)
with panda.create_context(frequency=1000, max_runtime=LEN) as ctx:
  while ctx.ok():
    ctrl.set_control(q[i], dq[i])
    i += 1

