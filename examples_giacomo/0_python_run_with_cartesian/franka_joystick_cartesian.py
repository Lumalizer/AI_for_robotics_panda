import numpy as np
import time
import os
from  scipy.spatial.transform import Rotation 
from panda_py import constants
constants.JOINT_LIMITS_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
constants.JOINT_LIMITS_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
import panda_py  # https://github.com/JeanElsner/panda-py
from panda_py import libfranka
from panda_py import controllers

import pyspacemouse #pip install pyspacemouse ; see instructions on https://pypi.org/project/pyspacemouse/


conversion_factor = 0.002
angle_conversion_factor = 0.8


success = pyspacemouse.open()#dof_callback=pyspacemouse.print_state, button_callback=pyspacemouse.print_buttons)
if not success:
    exit()


panda = panda_py.Panda("172.16.0.2")
gripper = libfranka.Gripper("172.16.0.2")

panda.move_to_start()
#gripper.homing()

pose = panda.get_pose()
x0 = pose[:3, 3]
q0 = Rotation.from_matrix(pose[:3, :3]).as_quat()


ctrl = controllers.CartesianImpedance()
panda.start_controller(ctrl)
last_time = time.time()
with panda.create_context(frequency=1e2, max_runtime=-1) as ctx:
    while ctx.ok():
        state = pyspacemouse.read()
        # state.x y z yaw pitch roll   state.button list [0,0]
        x, y, z = state.x, state.y, state.z
        roll, pitch, yaw = state.roll, state.pitch, state.yaw

        if abs(x)<0.05:
            x = 0
        if abs(y)<0.05:
            y = 0
        if abs(z)<0.05:
            z = 0
        
        if abs(roll)<0.05:
            roll = 0
        if abs(pitch)<0.05:
            pitch = 0
        if abs(yaw)<0.05:
            yaw = 0

        x0[0] += -y * conversion_factor # in meters ;   todo: suggested:  clip xyz to sy 20-30 cm box;  coords to be calculated
        x0[1] += x * conversion_factor
        x0[2] += z * conversion_factor

        rot = Rotation.from_quat(q0)
        delta_rot = Rotation.from_euler('zyx', np.array([yaw, pitch, -roll])*angle_conversion_factor, degrees=True)
        rot = rot * delta_rot
        q0 = rot.as_quat()

        #q = panda_py.ik(pose)    # opposite is pose = panda_py.fk(q)
        #q = q.clip(constants.JOINT_LIMITS_LOWER, constants.JOINT_LIMITS_UPPER)

        print(x0*100, state.x)

        ctrl.set_control(x0, q0)

        # aim at 100hz control
        print(time.time()-last_time)
        while time.time()-last_time < 1.0/200.0:
            time.sleep(0.01)
        last_time = time.time()

###  mouse -> robot frame
"""   lookijg at the robot from the front
x -> y
-y -> x
z -> z

yaw/pitch/roll
 


panda.enable_logging(2*1000) # 2s
[...]
panda.disable_logging()
log = panda.get_log()



"""




exit()










"""
Franka joystick moves:
    axis0: left stick (left-right) -> gripper open close (right=close)                                                               ->   -1 open to 1 close
    axis1: left stick (top-bottom) -> dz pos                                                                                         ->   -1 top to 1 bottom?
    axis2: right stick (left-right) -> dx pos (left=left as seen when looking AT the robot)                                          ->  -1 left to 1 right
    axis3: right stick (top-bottom) -> dy pos (up=move in depth away from the person facing the robot, i.e., toward the robot base)  -> -1 top to 1 bottom

TODO: try setting cartesian velocities instead of positions
"""


panda = panda_py.Panda("172.16.0.2")
gripper = libfranka.Gripper("172.16.0.2")
gripper_info = gripper.read_once()

panda.move_to_start()
#gripper.move(gripper_info.max_width, 0.03)
#gripper.grasp(0, 0.03, 50, 0.1, 0.1)
gripper.homing()


pose = panda.get_pose()
pose[0, 3] -= 0.01
pose[2, 3] -= 0.01
try:
    panda.move_to_pose(pose, speed_factor=0.1)
except RuntimeError:
    print('exception')
    panda.recover()
time.sleep(1)#gripper.move(gripper_info.max_width, 0.03)
#gripper.grasp(0, 0.03, 50, 0.1, 0.1)



q0 = panda.get_orientation()

#panda.teaching_mode(True)
ctrl = controllers.CartesianImpedance()

# current gripper width = gripper.read_once().width

m_per_step = 0.2 / 100.0

panda.start_controller(ctrl)
with panda.create_context(frequency=1e3) as ctx:
    while ctx.ok():
        pygame.event.pump()

        dx = joystick.get_axis(2) * m_per_step
        dy = joystick.get_axis(3) * m_per_step
        dz = joystick.get_axis(1) * m_per_step

        pose[0, 3] += dy
        pose[1, 3] += dx
        pose[2, 3] -= dz

        ctrl.set_control(pose[:3,3], q0)

        if joystick.get_button(1):
            gripper.grasp(0, 0.03, 50, 0.05, 0.05)
        elif joystick.get_button(4):
            gripper.move(gripper_info.max_width, 0.03)
        print(pose[:,3])

        time.sleep(0.01)



import time
st=time.time()
while time.time()-st<20:
    a,b,c = joystick.get_axis(0), joystick.get_axis(1), joystick.get_axis(2)
    print(a,'\t',b,'\t',c)
    time.sleep(0.1)