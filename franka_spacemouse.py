import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation
from panda_py import constants
constants.JOINT_LIMITS_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
constants.JOINT_LIMITS_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
import panda_py
from panda_py import libfranka
from spacemousecontroller import SpaceMouseController, SpaceMouseState
from panda_py import controllers
import datetime
import os
import pickle
import pandas as pd
from camera import Camera
import cv2

class FrankaController:
    def __init__(self, conversion_factor=0.003, angle_conversion_factor=0.4, mouse_axes_conversion=SpaceMouseState(1, 1, 1, 1, 1, 1), max_runtime=-1):
        self.panda = panda_py.Panda("172.16.0.2")
        self.gripper = libfranka.Gripper("172.16.0.2")
        
        self.spacemouse_controller = SpaceMouseController(button_callback=self.button_callback)
        self.mouse_axes_conversion = mouse_axes_conversion
        
        self.conversion_factor = conversion_factor
        self.angle_conversion_factor = angle_conversion_factor
        self.rotation_enabled = True
        self.is_gripping = False
        self.max_runtime = max_runtime
        self.is_recording = threading.Event()
        self._camera_logs = []
        self._camera_time = []
        self.camera = None

        self.reset_robot_position()
        self.ctrl = controllers.CartesianImpedance()

    def reset_robot_position(self):
        self.panda.move_to_start()
        time.sleep(1)

    def button_callback(self, state, buttons):
        if buttons[0]:  # left button
            print("Button 1 pressed")
            if self.is_recording.is_set():
                self.is_recording.clear()
            else:
                self.is_recording.set()
                print('Recording trajectory...')
        
        if buttons[1]:  # right button
            print("Button 2 pressed")
            self.is_gripping = not self.is_gripping  # flip early to log correctly

            if self.is_gripping:
                self.gripper.grasp(0, 0.2, 50, 0.04, 0.04)  # grip
            else:
                self.gripper.move(0.08, 0.2)  # release gripper

    def camera_thread_fn(self):
        while True:
            while self.is_recording.is_set():
                self._camera_logs.append(self.camera.get_frame())
                self._camera_time.append(time.time_ns())
            time.sleep(0.001)

    def start_camera_thread(self):
        self.camera = Camera()
        cam_thread = threading.Thread(target=self.camera_thread_fn, daemon=True)
        cam_thread.start()
    
    def process_log(self):
        logs = self.panda.get_log()

        # t = np.squeeze(t)
    
        q = logs['q']
        dq = logs['dq']
        t = np.array(logs['time'])
        t = (t-t[0]) / 1e3
        poses = [panda_py.fk(qq) for qq in q]

        gripper = np.array(self._logs['gripper'])
        gripper_time = np.array(self._logs['time'])

        cam_time = np.array(self._camera_time)
 
        cam_time = (cam_time - gripper_time[0]) / 1e9
        gripper_time = (gripper_time - gripper_time[0]) / 1e9

        # print(np.array(q).shape, np.array(dq).shape, np.array(poses).shape)
        return {'franka_t':t, 'franka_q':np.array(q), 'franka_dq':np.array(dq), 'franka_pose':np.array(poses), 'gripper_t':gripper_time, 'gripper_status':gripper, 'camera_frame_t':cam_time}

    def enter_logging(self):
        # logs directly from libfranka
        seconds_to_log = (self.max_runtime if self.max_runtime > 0 else 600)
        self.panda.enable_logging(buffer_size=seconds_to_log * 1000)

        # our own logs (gripper / camera)
        self._logs = {'gripper': [0], 'time': [time.time_ns()]}
        self._camera_logs = []
        self._camera_time = []
        
        self.is_recording.set()

    def exit_logging(self):
        self.is_recording.clear()
        
        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join("logs", "trajectory_"+str(date))
        camera_path = os.path.join(path, "camera")

        os.makedirs(path, exist_ok=True)

        self.panda.disable_logging()
        print('Trajectory recorded!')

        with open(os.path.join(path, 'trajectory.pkl'), 'wb') as f:
            pickle.dump(self.process_log(), f)

        print("Camera frames:" , len(self._camera_logs))
        os.makedirs(camera_path, exist_ok=True)
        
        # write mp4 video
        frame_height, frame_width = self._camera_logs[0].shape[:2]
        video_path = os.path.join(camera_path, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (frame_width, frame_height))

        for i, frame in enumerate(self._camera_logs):
            out.write(frame)
            
        out.release()
        
        print("Gripper frames", len(self._logs['gripper']))
        print("Gripper frames closed", sum(self._logs['gripper']))

    def enable_spacemouse_control(self, log=False):
        self.is_gripping = self.gripper.read_once().is_grasped
        self.pose = self.panda.get_pose()
        self.pos = self.pose[:3, 3]
        self.angles = Rotation.from_matrix(self.pose[:3, :3]).as_quat()

        print(f"Starting SpaceMouse control for {self.max_runtime} seconds...")
        self.panda.start_controller(self.ctrl)

        if log:
            self.enter_logging()
        
        with self.panda.create_context(frequency=1e2, max_runtime=self.max_runtime) as ctx:
            print('entering ctx')
            while ctx.ok():
                self._logs['gripper'].append(self.is_gripping)
                self._logs['time'].append(time.time_ns())

                mouse = self.spacemouse_controller.read()
                mouse = mouse * self.mouse_axes_conversion

                self.pos[0] += -mouse.y * self.conversion_factor
                self.pos[1] += mouse.x * self.conversion_factor
                self.pos[2] += mouse.z * self.conversion_factor

                rot = Rotation.from_quat(self.angles)
                new_d_rot = np.array([mouse.yaw, mouse.pitch, -mouse.roll]) * self.angle_conversion_factor * int(self.rotation_enabled)
                delta_rot = Rotation.from_euler('zyx', new_d_rot, degrees=True)
                rot = rot * delta_rot
                self.angles = rot.as_quat()

                self.ctrl.set_control(self.pos, self.angles)

                if not self.is_recording.is_set():
                    self.panda.stop_controller()
                    break
        
        if log:
            self.exit_logging()

        self.reset_robot_position()
        time.sleep(2)

    def collect_demonstrations(self, quantity=10):
        for i in range(quantity):
            print(f"Collecting demonstration {i + 1} of {quantity}...")
            self.enable_spacemouse_control(log=True)


if __name__ == "__main__":
    fc = FrankaController(max_runtime=-1)

    try:
        fc.start_camera_thread()
    except Exception as e:
        print("Camera not connected.")
        exit()

    while True:
        try:
            fc.spacemouse_controller.read()
            time.sleep(0.001)

            if fc.is_recording.is_set():
                fc.enable_spacemouse_control(log=True)
        except Exception as e:
            print(f"Error in recording trajectory: {e}")
            fc.spacemouse_controller.close()
            time.sleep(1)
            fc.panda.stop_controller()
            fc.panda.disable_logging()
            
            raise e