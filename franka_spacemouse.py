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
from camera import Camera

class FrankaController:
    def __init__(self, 
                 logger:'Logger'=None, 
                 conversion_factor=0.003, 
                 angle_conversion_factor=0.4, 
                 mouse_axes_conversion=SpaceMouseState(1, 1, 1, 1, 1, 1), 
                 max_runtime=-1):
        
        self.panda = panda_py.Panda("172.16.0.2")
        self.gripper = libfranka.Gripper("172.16.0.2")
        
        self.spacemouse_controller = SpaceMouseController(self.button_callback, conversion_factor, angle_conversion_factor, mouse_axes_conversion)
        
        self.is_gripping = False
        self.max_runtime = max_runtime
        
        self.is_recording = threading.Event()
        self.camera = None
        if logger is None:
            self.logger = Logger(self)

        self.reset_robot_position()
        self.ctrl = controllers.CartesianImpedance()

    def reset_robot_position(self):
        self.panda.move_to_start()
        time.sleep(1)

    def button_callback(self, state, buttons):
        if buttons[0]:  # left button
            self.toggle_recording()
        
        if buttons[1]:  # right button
            self.toggle_gripper() 

    def toggle_gripper(self):
        self.is_gripping = not self.is_gripping  # flip early to log correctly

        if self.is_gripping:
            self.gripper.grasp(0, 0.2, 50, 0.04, 0.04)  # grip
        else:
            self.gripper.move(0.08, 0.2)

    def toggle_recording(self):
        if self.is_recording.is_set():
            self.is_recording.clear()
        else:
            self.is_recording.set()
            print('Recording trajectory...')

    def camera_thread_fn(self):
        while True:
            while self.is_recording.is_set():
                self.logger._camera_logs.append(self.camera.get_frame())
                self.logger._camera_time.append(time.time_ns())
            time.sleep(0.001)

    def _start_camera_thread(self):
        # started automatically by the logger
        try:
            self.camera = Camera()
            cam_thread = threading.Thread(target=self.camera_thread_fn, daemon=True)
            cam_thread.start()
        except Exception as e:
            print("Camera not connected.")
            raise e

    def enable_spacemouse_control(self, log=True):
        self.is_gripping = self.gripper.read_once().is_grasped
        self.pose = self.panda.get_pose()
        self.pos = self.pose[:3, 3]
        self.angles = Rotation.from_matrix(self.pose[:3, :3]).as_quat()

        print(f"Starting SpaceMouse control for {self.max_runtime} seconds...")
        self.panda.start_controller(self.ctrl)

        if log:
            self.logger.enter_logging()
        
        with self.panda.create_context(frequency=1e2, max_runtime=self.max_runtime) as ctx:
            while ctx.ok():
                self.logger._logs['gripper'].append(self.is_gripping)
                self.logger._logs['time'].append(time.time_ns())

                mouse = self.spacemouse_controller.read()

                self.pos[0] += -mouse.y
                self.pos[1] += mouse.x
                self.pos[2] += mouse.z

                rot = Rotation.from_quat(self.angles)
                new_d_rot = np.array([mouse.yaw, mouse.pitch, -mouse.roll])
                delta_rot = Rotation.from_euler('zyx', new_d_rot, degrees=True)
                rot = rot * delta_rot
                self.angles = rot.as_quat()

                self.ctrl.set_control(self.pos, self.angles)

                if not self.is_recording.is_set():
                    self.panda.stop_controller()
                    break
        
        if log:
            self.logger.exit_logging()

        self.reset_robot_position()
            
    def collect_demonstrations(self, amount=10):
        print(f"Press left button on the space mouse to start or stop recording a new trajectory. ({amount} remaining)")
        
        while amount:
            try:
                self.spacemouse_controller.read()
                time.sleep(0.001)

                if self.is_recording.is_set():
                    self.enable_spacemouse_control()
                    amount -= 1
                    
                    print(f"Press left button on the space mouse to start or stop recording a new trajectory. ({amount} remaining)")
                    
            except Exception as e:
                print(f"Error in recording trajectory: {e}")
                self.reset_robot_position()
                self.panda.stop_controller()
                self.logger.exit_logging(save=False)
                
                if type(e) == RuntimeError:
                    print("attempting to restart...\n\n")
                    print(f"Press left button on the space mouse to start or stop recording a new trajectory. ({amount} remaining)")
                    self.is_recording.clear()
                else:
                    raise e


if __name__ == "__main__":
    from logger import Logger
    
    fc = FrankaController()
    fc.collect_demonstrations()