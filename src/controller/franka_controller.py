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
from pynput import keyboard

class FrankaController:
    def __init__(self,
                 env:RealFrankaEnv=None,
                 logger:Logger=None, 
                 dataset_name="no_name",
                 
                 xyz_multiplier=0.03, 
                 angle_multiplier=15,
                 
                 step_duration_s=1/30,
                 execution_horizon=1,
                 
                 action_multiplier: int=1,
                 fps=30,
                 
                 mode="octo",
                 unnorm_key=None,
                 randomize_starting_position=False):
        
        self.env = env if env else RealFrankaEnv(step_duration_s=step_duration_s, multiplier=action_multiplier, action_space="cartesian")
        self.execution_horizon = execution_horizon
        
        if execution_horizon > 1:
            self.env = RHCWrapper(self.env, execution_horizon)

        self.is_gripping = False
        self.is_pre_controlling = threading.Event() # allows moving the arm before recording logs
        self.is_recording = threading.Event()
        
        self.logger = logger if logger else Logger(self, fps)
        self.dataset_name = dataset_name
        self.mode = mode
        
        self.step_duration_s = step_duration_s
        self.fps = fps
        self.randomize_starting_position = randomize_starting_position
        
        if mode not in ["octo", "openvla"]:
            raise ValueError(f"Invalid mode: {mode}. Must be one of ['octo', 'openvla']")
        
        if not unnorm_key:
            if mode == "octo":
                self.unnorm_key = "action"
            elif mode == "openvla":
                self.unnorm_key = "air_net"

        self.spacemouse_controller = SpaceMouseController(
            xyz_multiplier, angle_multiplier, 
            button_left_callback=self.toggle_recording, 
            button_right_callback=self.toggle_gripper)
        
        if not self.spacemouse_controller:
            print(f"SpaceMouse not connected.")

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

    def move_randomly(self, n_steps=25, factors=[0,0,0,1,2,3]):
        factor = random.choice(factors)

        if not factor:
            return

        # Random dx dy dz dyaw dpitch droll on reset
        dx = factor * np.random.uniform(-0.1, 0.3) / n_steps
        dy = factor * np.random.uniform(-0.3, 0.3) / n_steps
        dz = factor * np.random.uniform(-0.3, 0) / n_steps
        dyaw = factor * np.random.uniform(-15, 15) / n_steps
        dpitch = factor * np.random.uniform(-15, 15) / n_steps
        droll = factor * np.random.uniform(-45, 45) / n_steps

        for _ in range(n_steps):
            self.env.step(np.array([dx, dy, dz, dyaw, dpitch, droll, 0]))

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
            time.sleep(1) # TODO: maybe remove this
            recorded_successfully = self.logger.exit_logging()
        
        if release_gripper_on_exit and self.is_gripping:
            self.toggle_gripper()
        
        if reset:
            self.env.reset()
            if self.randomize_starting_position:
                self.move_randomly()
        
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

    @staticmethod         
    def to_base64(img: np.ndarray) -> str:
        return base64.b64encode(zlib.compress(img.tobytes())).decode('utf-8')
    
    @staticmethod
    def resize_image(img: np.ndarray, size: int=256) -> np.ndarray:
        return cv2.resize(img, (size, size))
    
    def get_current_state_for_inference(self, add_proprio: bool = False) -> tuple[dict, np.ndarray, np.ndarray]:
        data = {"unnorm_key": self.unnorm_key}
        
        if add_proprio:
            state = self.env.get_state()
            
            # 7 joint angles, 3 xyz, 1 gripper
            state = np.concatenate((state[:10], np.expand_dims(state[21], axis=0)))
            state = np.expand_dims(state, axis=0)
            state = np.expand_dims(state, axis=0)

            data['proprio'] = state
        
        cam_primary, cam_wrist = self.logger.primary_camera, self.logger.wrist_camera
        if cam_primary:
            img_primary = cam_primary.logs[-1].copy()
            img_primary = self.resize_image(img_primary)
            img_primary = cv2.cvtColor(img_primary, cv2.COLOR_BGR2RGB)
            img_key = "image" if self.mode == "openvla" else "primary_image"
            data[img_key] = self.to_base64(img_primary)
        if cam_wrist:
            img_wrist = cam_wrist.logs[-1].copy()
            img_wrist = self.resize_image(img_wrist, size=128)
            img_wrist = cv2.cvtColor(img_wrist, cv2.COLOR_BGR2RGB)
            if not self.mode == "openvla":
                data['wrist_image'] = self.to_base64(img_wrist)
        return data
                
    def run_from_server(self, ip: str="http://0.0.0.0:8000/act", instruction=None, save=False, evaluating=True, max_seconds=30):
        while not instruction:
            instruction = input(f"Enter instruction (or keep empty to ({instruction}): ")

        key_pressed = False
        
        def on_press(key):
            nonlocal key_pressed
            try:
                if key.char == ("s"):
                    key_pressed = 's'
                elif key.char == ("f"):
                    key_pressed = 'f'
                elif key.char == ("r"):
                    key_pressed = 'r'
            except AttributeError:
                pass
             
            
        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        self.env.reset()
        
        if self.mode == "openvla":
            self.move_randomly()
            
        self.logger.enter_logging()
        for camera in self.logger.cameras:
            while not camera.logs:
                time.sleep(0.1)
        
        print(f"Performing command ({instruction}) for {max_seconds} seconds...")
        start = time.time()
        while True:
            time_elapsed = round(time.time() - start, 3)
            if (time_elapsed) > max_seconds:
                break
            
            if key_pressed == 's':
                print(f"Success... {time_elapsed}s")
                break
            elif key_pressed == 'f':
                print(f"Failure... {time_elapsed}s")
                break
            elif key_pressed == 'r':
                print(f"Reset... {time_elapsed}s")
                break
                

            state = self.get_current_state_for_inference(add_proprio=False)
            state['instruction'] = instruction
                
            try:
                action = requests.post(ip, json=state).json()
                
                logged_action = action.copy() if self.mode == "openvla" else action[0].copy()
                action = action[0] if self.mode == "octo" and self.execution_horizon == 1 else action
                
                self.logger.log(logged_action, inference=True)
                self.env.step(action)
            except Exception as e:
                print(f"Error in server communication: {e}")
                break
    
        self.env.stop_controller()
        listener.stop()
        self.logger.exit_logging(save=False, inference=True, task_desc=instruction)
        
        if not key_pressed and evaluating:
            while key_pressed not in ['s', 'f', 'r']:
                key_pressed = input("Enter 's' for success, 'f' for failure, 'r' for reset: ")
        
        if key_pressed == 'r':
            return None
        
        self.logger.exit_logging(save=save, inference=True, task_desc=instruction, 
                                 success_or_failure=(key_pressed == 's'), total_time=time_elapsed)
        
        
    def continually_run_from_server(self, instruction: str = "pick up the blue block", save=False, max_seconds=30):      
        while True:
            self.run_from_server(instruction=instruction, save=save, max_seconds=max_seconds), 
