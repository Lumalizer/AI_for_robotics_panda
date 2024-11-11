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
import time

class FrankaController:
    def __init__(self,
                 env:RealFrankaEnv=None,
                 logger:Logger=None, 
                 dataset_name="no_name",
                 
                 conversion_factor=0.03, 
                 angle_conversion_factor=15,
                 mouse_axes_conversion=SpaceMouseState(1, 1, 1, 1, 1, 1),
                 
                 step_duration_s=0.2,
                 step_duration_s_spacemouse=0.01,
                 
                 fps=30,
                 max_runtime=-1):
        
        self.env = env if env else RealFrankaEnv(step_duration_s=step_duration_s, action_space="cartesian")
        
        self.is_gripping = False
        self.is_pre_controlling = threading.Event() # allows moving the arm before recording logs
        self.is_recording = threading.Event()
        
        self.logger = logger if logger else Logger(self, fps)
        self.dataset_name = dataset_name
        
        self.step_duration_s = step_duration_s
        self.step_duration_s_spacemouse = step_duration_s_spacemouse
        
        self.max_runtime = max_runtime
        self.fps = fps
        
        # space mouse
        try:
            self.spacemouse_controller = SpaceMouseController(
                conversion_factor, angle_conversion_factor, mouse_axes_conversion, 
                button_left_callback=self.toggle_recording, button_right_callback=self.toggle_gripper)
        except Exception as e:
            self.spacemouse_controller = None
            print("SpaceMouse not connected.")
        

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
        img = self.logger.camera.logs[-1]
        img = np.expand_dims(img, axis=0)
        
        state = self.env.get_state()
        state = np.expand_dims(state, axis=0)
        
        return {'proprio': state, 'image_primary': img}   

    def enable_spacemouse_control(self, event: threading.Event, log=True, release_gripper_on_exit=True, reset=True):
        # print(f"Starting SpaceMouse control for {self.max_runtime} seconds...")
        self.env.step_duration_s = self.step_duration_s_spacemouse
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
            
        self.env.step_duration_s = self.step_duration_s
        
        if reset:
            self.env.reset()
        
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
                
    def run_from_server(self, ip:str="http://0.0.0.0:8000/act"):
        instruction = input("Enter instruction (or keep empty to 'grasp the blue block'): ")
        if not instruction:
            instruction = "grasp the blue block"
        
        self.logger.enter_logging()
        while not self.logger.camera.logs:
            time.sleep(0.1)
        
        while True:
            state = self.get_current_state_for_inference()
            img = base64.b64encode(zlib.compress(state['image_primary'].tobytes())).decode('utf-8')
            
            try:
                action = requests.post(ip, json={"image": img, "instruction": instruction}).json()
                self.env.step(action)
            except Exception as e:
                print(f"Error in server communication: {e}")
                break
        
        self.logger.exit_logging(save=False)
