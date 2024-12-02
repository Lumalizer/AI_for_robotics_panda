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
import cv2
from controller.wrappers_octo import RHCWrapper
import random

class FrankaController:
    def __init__(self,
                 env:RealFrankaEnv=None,
                 logger:Logger=None, 
                 dataset_name="no_name",
                 mode: str="demonstration",
                 
                 conversion_factor=0.03, 
                 angle_conversion_factor=15,
                 mouse_axes_conversion=SpaceMouseState(1, 1, 1, 1, 1, 1),
                 
                 step_duration_s=1/30,
                 receding_horizon=None,
                 
                 fps=30,
                 
                 randomize_starting_position=False):
        
        if mode not in ['demonstration', 'openvla', 'octo']:
            raise ValueError(f"Invalid mode: {mode}. Must be one of ['demonstration', 'openvla', 'octo']")
    
        if mode == 'openvla':
            multiplier = 2
        elif mode == 'octo':
            multiplier = 0.02
        elif mode == 'demonstration':
            multiplier = 1
        
        self.env = env if env else RealFrankaEnv(step_duration_s=step_duration_s, multiplier=multiplier, action_space="cartesian")
        if receding_horizon:
            self.env = RHCWrapper(self.env, receding_horizon)

        self.is_gripping = False
        self.is_pre_controlling = threading.Event() # allows moving the arm before recording logs
        self.is_recording = threading.Event()
        
        self.logger = logger if logger else Logger(self, fps)
        self.dataset_name = dataset_name
        
        self.step_duration_s = step_duration_s
        
        self.fps = fps
        
        self.randomize_starting_position = randomize_starting_position

        # space mouse
        try:
            self.spacemouse_controller = SpaceMouseController(
                conversion_factor, angle_conversion_factor, mouse_axes_conversion, 
                button_left_callback=self.toggle_recording, button_right_callback=self.toggle_gripper)
        except Exception as e:
            self.spacemouse_controller = None
            print(f"SpaceMouse not connected. {e}")
        

        self.env.reset()

    def toggle_gripper(self):
        self.is_gripping = not self.is_gripping  # flip early to log correctly

        if self.is_gripping:
            self.env.close_gripper()
        else:
            self.env.open_gripper()

    def toggle_recording(self):
        self.is_pre_controlling.clear()
        
        if self.is_recording.is_set():
            self.env.stop_controller()
            self.is_recording.clear()
        else:
            self.is_recording.set()
            print('Recording trajectory............\n')
        
    def get_current_state_for_inference(self) -> tuple[dict, np.ndarray, np.ndarray]:
        state = self.env.get_state()
        
        # hack for octo
        state = np.concatenate((state[:10], np.expand_dims(state[21], axis=0)))
        state = np.expand_dims(state, axis=0)
        # end hack
        
        state = np.expand_dims(state, axis=0)
        data = {'proprio': state}
        
        cam_primary, cam_wrist = self.logger.primary_camera, self.logger.wrist_camera
        if cam_primary:
            img_primary = cam_primary.logs[-1]
            img_primary = np.expand_dims(img_primary, axis=0)
            data['primary_image'] = img_primary
        if cam_wrist:
            img_wrist = cam_wrist.logs[-1]
            img_wrist = np.expand_dims(img_wrist, axis=0)
            data['wrist_image'] = img_wrist
        
        return data

    def enable_spacemouse_control(self, event: threading.Event, log=True, release_gripper_on_exit=True, reset=True):
        # print(f"Starting SpaceMouse control for {self.max_runtime} seconds...")
        recorded_successfully = False
        
        log and self.logger.enter_logging()        
        
        while event.is_set():
            mouse = self.spacemouse_controller.read() 
            action = np.array([*mouse.delta_pos, *mouse.delta_rot, self.is_gripping])
            
            self.logger.log(action)
            self.env.step(action)

        if log:
            time.sleep(1)
            recorded_successfully = self.logger.exit_logging()
        
        if release_gripper_on_exit and self.is_gripping:
            self.toggle_gripper()
        
        if reset:
            self.env.reset()
            if self.randomize_starting_position:
                n_steps = 25
                
                factor = random.choice([0, 0, 0, 1, 2, 3])

                # Random dx dy dz dyaw dpitch droll on reset
                dx = factor * np.random.uniform(-0.1, 0.3) / n_steps
                dy = factor * np.random.uniform(-0.3, 0.3) / n_steps
                dz = factor * np.random.uniform(-0.3, 0) / n_steps
                dyaw = factor * np.random.uniform(-15, 15) / n_steps
                dpitch = factor * np.random.uniform(-15, 15) / n_steps
                droll = factor * np.random.uniform(-45, 45) / n_steps

                for s in range(n_steps):
                    self.env.step(np.array([dx, dy, dz, dyaw, dpitch, droll, 0]))
        
        return recorded_successfully
            
    def collect_demonstrations(self, amount=10, enable_pre_control=True):
        while amount:
            try:
                self.spacemouse_controller.read()

                # pre-control before recording
                if enable_pre_control:
                    self.is_pre_controlling.set()
                    print(f"\033[93mPre-control enabled.\033[0m Move the arm into desired recording start position. \nPress \033[92mleft button\033[0m on the space mouse to start or stop recording a new trajectory. ({amount} remaining)")
                    self.enable_spacemouse_control(self.is_pre_controlling, log=False, release_gripper_on_exit=False, reset=False)
                else:
                    print(f"Press \033[92mleft button\033[0m on the space mouse to start or stop recording a new trajectory. ({amount} remaining)")
                
                time.sleep(0.001)

                # recording
                if self.is_recording.is_set():
                    if self.enable_spacemouse_control(self.is_recording):
                        amount -= 1
                    
            except Exception as e:
                print(f"Error in recording trajectory: {e}")
                self.env.reset()
                self.logger.exit_logging(save=False)
                
                if type(e) == RuntimeError:
                    print("attempting to restart...\n\n")
                    self.is_recording.clear()
                else:
                    raise e
                
    def to_base64(self, img:np.ndarray) -> str:
        return base64.b64encode(zlib.compress(img.tobytes())).decode('utf-8')
    
    def resize_image(self, img:np.ndarray, size:int=256) -> np.ndarray:
        return cv2.resize(img, (size, size))
                
    def run_from_server(self, ip:str="http://0.0.0.0:8000/act", instruction=None, max_seconds=20):
        if not instruction:
            instruction = input(f"Enter instruction (or keep empty to ({instruction}): ")

        self.logger.enter_logging()
        for camera in self.logger.cameras:
            while not camera.logs:
                time.sleep(0.1)
                
        self.env.reset()
        
        print(f"Performing command ({instruction}) for {max_seconds} seconds...")
        start = time.time()
        while True:
            if (time.time() - start) > max_seconds:
                break
            state = self.get_current_state_for_inference()
            # data = {"instruction": instruction, "state": state['proprio']}
            data = {"instruction": instruction}
            
            if state["primary_image"] is not None:
                img1 = state['primary_image']
                img1 = self.resize_image(img1[0])
                data["primary_image"] = self.to_base64(img1)
            if state["wrist_image"] is not None:
                img2 = state['wrist_image']
                img2 = self.resize_image(img2[0], size=128)
                data["wrist_image"] = self.to_base64(img2)
                
            try:
                action = requests.post(ip, json=data).json()
                self.env.step(action)
            except Exception as e:
                print(f"Error in server communication: {e}")
                break
    
        self.logger.exit_logging(save=False)
        
    def continually_run_from_server(self, instruction: str = "pick up the blue block"):      
        while True:
            self.run_from_server(instruction=instruction)
