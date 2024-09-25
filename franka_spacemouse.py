import numpy as np
import time
from  scipy.spatial.transform import Rotation
from panda_py import constants
constants.JOINT_LIMITS_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
constants.JOINT_LIMITS_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
import panda_py
from panda_py import libfranka
from spacemousecontroller import SpaceMouseController
from panda_py import controllers

class FrankaController:
    def __init__(self):
        self.panda = panda_py.Panda("172.16.0.2")
        self.gripper = libfranka.Gripper("172.16.0.2")
        self.spacemouse_controller = SpaceMouseController()
        
        self.conversion_factor = 0.002
        self.angle_conversion_factor = 0.8

        self.panda.move_to_start()
        time.sleep(1)

        self.pose = self.panda.get_pose()
        self.x0 = self.pose[:3, 3]
        self.q0 = Rotation.from_matrix(self.pose[:3, :3]).as_quat()

        self.ctrl = controllers.CartesianImpedance()


    def enable_spacemouse_control(self):
        self.panda.start_controller(self.ctrl)

        with self.panda.create_context(frequency=1e2, max_runtime=-1) as ctx:
            while ctx.ok():
                mouse = self.spacemouse_controller.read()

                self.x0[0] += -mouse.y * self.conversion_factor # in meters ;   todo: suggested:  clip xyz to sy 20-30 cm box;  coords to be calculated
                self.x0[1] += mouse.x * self.conversion_factor
                self.x0[2] += mouse.z * self.conversion_factor

                rot = Rotation.from_quat(self.q0)
                delta_rot = Rotation.from_euler('zyx', np.array([mouse.yaw, mouse.pitch, -mouse.roll])*self.angle_conversion_factor, degrees=True)
                rot = rot * delta_rot
                self.q0 = rot.as_quat()

                #q = panda_py.ik(pose)    # opposite is pose = panda_py.fk(q)
                #q = q.clip(constants.JOINT_LIMITS_LOWER, constants.JOINT_LIMITS_UPPER)

                print(self.x0*100, mouse.x)

                self.ctrl.set_control(self.x0, self.q0)

if __name__ == "__main__":
    fc = FrankaController()
    fc.enable_spacemouse_control()