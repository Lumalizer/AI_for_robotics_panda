import threading
import time
import numpy as np
import json_numpy
json_numpy.patch()
from in_out.spacemousecontroller import SpaceMouseController, SpaceMouseState
from in_out.logger import Logger
import threading
import requests
from controller.realfranka_env import RealFrankaEnv
import base64
import zlib

class FrankaController:
    def __init__(self,
                 env:RealFrankaEnv=None,
                 logger:Logger=None, 
                 conversion_factor=0.03, 
                 angle_conversion_factor=25,
                 step_duration_s=0.2,
                 step_duration_s_spacemouse=0.01,
                 mouse_axes_conversion=SpaceMouseState(1, 1, 1, 1, 1, 1), 
                 dataset_name="no_name",
                 max_runtime=-1):
        
        self.step_duration_s = step_duration_s
        self.step_duration_s_spacemouse = step_duration_s_spacemouse
        self.env = RealFrankaEnv(step_duration_s=step_duration_s, action_space="cartesian")
        
        self.dataset_name = dataset_name
        self.max_runtime = max_runtime
        
        # space mouse
        try:
            self.spacemouse_controller = SpaceMouseController(self.button_callback, conversion_factor, angle_conversion_factor, mouse_axes_conversion)
        except Exception as e:
            self.spacemouse_controller = None
            print("SpaceMouse not connected.")
        
        # logs / recording
        self.is_gripping = False
        self.is_recording = threading.Event()
        
        if logger is None:
            self.logger = Logger(self)

        self.env.reset()

    def button_callback(self, state, buttons):
        if buttons[0]:  # left button
            self.toggle_recording()
        
        if buttons[1]:  # right button
            self.toggle_gripper() 

    def toggle_gripper(self):
        self.is_gripping = not self.is_gripping  # flip early to log correctly

        if self.is_gripping:
            self.env.close_gripper()
        else:
            self.env.open_gripper()

    def toggle_recording(self):
        if self.is_recording.is_set():
            self.env.stop_controller()
            self.is_recording.clear()
        else:
            self.is_recording.set()
            print('Recording trajectory...')
        
    
    def get_current_state_for_inference(self) -> tuple[dict, np.ndarray, np.ndarray]:
        # gripper_status = np.array([self.gripper.read_once().is_grasped])
        # TODO fix gripper blocking
        mask = np.array([1])
        
        img = self.logger.get_camera_frame_resized()
        img = np.expand_dims(img, axis=0)
        
        state = self.env.get_state()
        state = np.expand_dims(state, axis=0)
        
        return {'proprio': state, 'image_primary': img, 'timestep_pad_mask': mask}   

    def enable_spacemouse_control(self, log=True, release_gripper_on_exit=True):
        print(f"Starting SpaceMouse control for {self.max_runtime} seconds...")
        self.env.step_duration_s = self.step_duration_s_spacemouse
        
        log and self.logger.enter_logging()        
        
        while self.is_recording.is_set():
            mouse = self.spacemouse_controller.read()

            delta_pos = np.array([-mouse.y, mouse.x, mouse.z])
            delta_rot = np.array([mouse.yaw, mouse.pitch, -mouse.roll])
            
            action = np.array([*delta_pos, *delta_rot, self.is_gripping])
            
            self.logger.log_action(action)
            self.logger.log_gripper()
            self.env.step(action)

        log and self.logger.exit_logging()
        
        if release_gripper_on_exit and self.is_gripping:
            self.toggle_gripper()
            
        self.env.step_duration_s = self.step_duration_s
        self.env.reset()
            
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
                self.env.reset()
                self.logger.exit_logging(save=False)
                
                if type(e) == RuntimeError:
                    print("attempting to restart...\n\n")
                    print(f"Press left button on the space mouse to start or stop recording a new trajectory. ({amount} remaining)")
                    self.is_recording.clear()
                else:
                    raise e
                
    def run_from_server(self, ip:str="http://0.0.0.0:8000/act"):
        instruction = input("Enter instruction (or keep empty to 'grasp the blue block'): ")
        if not instruction:
            instruction = "grasp the blue block"
        
        self.logger.enter_logging()
        while not self.logger._camera_logs:
            time.sleep(0.1)
        
        while True:
            state = self.get_current_state_for_inference()
            img = base64.b64encode(zlib.compress(state['image_primary'].tobytes())).decode('utf-8')
            
            action = requests.post(ip, json={"image": img, "instruction": instruction}).json()
            self.env.step(action)
        
        self.logger.exit_logging(save=False)
