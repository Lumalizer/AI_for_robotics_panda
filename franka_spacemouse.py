import numpy as np
import time
from  scipy.spatial.transform import Rotation
from panda_py import constants
constants.JOINT_LIMITS_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
constants.JOINT_LIMITS_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
import panda_py
from panda_py import libfranka
from spacemousecontroller import SpaceMouseController, SpaceMouseState
from panda_py import controllers
from pynput import keyboard
import datetime
import os
import pickle
import pandas as pd
from camera import Camera
import threading 

class FrankaController:
    def __init__(self, conversion_factor=0.002, angle_conversion_factor=0.8, mouse_axes_conversion=SpaceMouseState(1, 1, 1, 1, 1, 1), max_runtime=-1):
        self.panda = panda_py.Panda("172.16.0.2")
        self.gripper = libfranka.Gripper("172.16.0.2")

        try:
            self.camera = Camera()
        except:
            print("Camera not connected.")
            self.camera = None

        self.spacemouse_controller = SpaceMouseController(button_callback=self.button_callback)
        self.mouse_axes_conversion = mouse_axes_conversion
        
        self.conversion_factor = conversion_factor
        self.angle_conversion_factor = angle_conversion_factor
        self.rotation_enabled = True
        self.is_gripping = False
        self.max_runtime = max_runtime
        
        # threading
        self._logs = {'gripper': [], 'time': []}
        self._camera_logs = []
        self._logging_thread = None
        self._stop_logging = threading.Event()

        self.reset_robot_position()

        self.ctrl = controllers.CartesianImpedance()

    def reset_robot_position(self):
        self.panda.move_to_start()
        time.sleep(1)

    def button_callback(self, state, buttons):
        if buttons[0]: # left button
            print("Button 1 pressed")
            self.rotation_enabled = not self.rotation_enabled
        
        if buttons[1]: # right button
            print("Button 2 pressed")
            
            self.is_gripping = not self.is_gripping # flip early to log correctly

            if self.is_gripping:
                self.gripper.grasp(0, 0.2, 50, 0.04, 0.04) # grip
            else:
                self.gripper.move(0.08, 0.2) # release gripper


    def process_log(self):
        logs = self.panda.get_log()

        q = logs['q']
        dq = logs['dq']
        t = logs['time']
        poses = [panda_py.fk(qq) for qq in q]

        # print(np.array(q).shape, np.array(dq).shape, np.array(poses).shape)
        return [t, np.array(q), np.array(dq), np.array(poses)]

    def enter_logging(self):
        
        # logging thread
        self._stop_logging.clear()
        self._logging_thread = threading.Thread(target=self._log_thread)
        self._logging_thread.start()
        print('logging thread started')

        # logs directly from libfranka
        print('enabling libfranka logging')
        seconds_to_log = (self.max_runtime if self.max_runtime > 0 else 10)
        self.panda.enable_logging(buffer_size=seconds_to_log * 1000)

        # our own logs (gripper / camera)
        self._logs = {'gripper': [], 'time': []}
        self._camera_logs = []

    def _log_thread(self):
        while not self._stop_logging.is_set():
            self.log_step()
            time.sleep(0.01)

    def log_step(self):
        self._logs['gripper'].append(self.is_gripping)
        self._logs['time'].append(time.time_ns())
        self._camera_logs.append(self.camera.get_frame())

    def exit_logging(self):
        self._stop_logging.set()
        if self._logging_thread is not None:
            self._logging_thread.join()

        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join("logs", "trajectory_"+str(date))
        camera_path = os.path.join(path, "camera")

        os.makedirs(path, exist_ok=True)

        self.panda.disable_logging()
        print('Trajectory recorded!')

        with open(os.path.join(path, 'trajectory.pkl'), 'wb') as f:
            pickle.dump(self.process_log(), f)

        pd.DataFrame(self._logs).to_csv(os.path.join(path, 'logs.csv'))

        if self.camera is not None:
            os.makedirs(camera_path, exist_ok=True)
            for i, frame in enumerate(self._camera_logs):
                frame.save(os.path.join(camera_path, f"{i}.png"))

        # print(len(self._logs['gripper']))
        # print(sum(self._logs['gripper']))
        print()

    def enable_spacemouse_control(self, log=False):
        self.is_gripping = self.gripper.read_once().is_grasped
        self.pose = self.panda.get_pose()
        self.pos = self.pose[:3, 3]
        self.angles = Rotation.from_matrix(self.pose[:3, :3]).as_quat()

        print(f"Starting SpaceMouse control for {self.max_runtime} seconds...")
        self.panda.start_controller(self.ctrl)

        log and self.enter_logging()
        
        with self.panda.create_context(frequency=1e3, max_runtime=self.max_runtime) as ctx:
            iteration = 0
            while ctx.ok():
                # log the gripper / camera time while collecting demonstrations -> check which demonstrations are lost

                # if iteration % 10 == 0: # collect grapper / camera logs every 10 iterations (10 per second)
                #     self._log_thread()

                iteration += 1

                mouse = self.spacemouse_controller.read()
                mouse = mouse * self.mouse_axes_conversion

                self.pos[0] += -mouse.y * self.conversion_factor # in meters ;   todo: suggested:  clip xyz to sy 20-30 cm box;  coords to be calculated
                self.pos[1] += mouse.x * self.conversion_factor
                self.pos[2] += mouse.z * self.conversion_factor

                rot = Rotation.from_quat(self.angles)
                new_d_rot = np.array([mouse.yaw, mouse.pitch, -mouse.roll]) * self.angle_conversion_factor * int(self.rotation_enabled)
                delta_rot = Rotation.from_euler('zyx', new_d_rot, degrees=True)
                rot = rot * delta_rot
                self.angles = rot.as_quat()

                #q = panda_py.ik(pose)    # opposite is pose = panda_py.fk(q)
                #q = q.clip(constants.JOINT_LIMITS_LOWER, constants.JOINT_LIMITS_UPPER)

                self.ctrl.set_control(self.pos, self.angles)
        
        self.reset_robot_position()
        self.panda.stop_controller()
        log and self.exit_logging()
        time.sleep(2)

    def collect_demonstrations(self, quantity = 10):
        for i in range(quantity):
            print(f"Collecting demonstration {i+1} of {quantity}...")
            self.enable_spacemouse_control(log=True)

if __name__ == "__main__":
    fc = FrankaController(max_runtime=5)
    fc.collect_demonstrations(1)