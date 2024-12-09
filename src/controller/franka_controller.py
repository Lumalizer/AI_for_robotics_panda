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
from options import Options
import termios
import sys

class FrankaController:
    def __init__(self, 
                 options: Options,
                 env:RealFrankaEnv=None,
                 logger:Logger=None):
        
        self.options = options
        
        if not env:
            env = RealFrankaEnv(step_duration_s=options.step_duration_s, 
                                multiplier=options.action_multiplier, action_space="cartesian")
        self.env = env
        
        if options.execution_horizon > 1:
            self.env = RHCWrapper(self.env, options.execution_horizon)

        self.is_gripping = False
        self.is_pre_controlling = threading.Event() # allows moving the arm before recording logs
        self.is_recording = threading.Event()
        
        self.logger = logger if logger else Logger(self, options.fps)


        self.spacemouse_controller = SpaceMouseController(
            options.xyz_multiplier, options.angle_multiplier, 
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

    def enable_spacemouse_control(self, event: threading.Event, release_gripper_on_exit=True, log=None, reset=True):
        # print(f"Starting SpaceMouse control for {self.max_runtime} seconds...")
        recorded_successfully = False
        
        if log is None:
            log = self.options.log
            
        log and self.logger.enter_logging()        
        
        while event.is_set():
            mouse = self.spacemouse_controller.read() 
            action = np.array([*mouse.delta_pos, *mouse.delta_rot, self.is_gripping])
            
            self.logger.log(action)
            self.env.step(action)

        if log:
            time.sleep(1) # TODO: maybe remove this
            recorded_successfully = self.logger.exit_logging(self.options)
        
        if release_gripper_on_exit and self.is_gripping:
            self.toggle_gripper()
        
        if reset:
            self.env.reset()
            if self.options.randomize_starting_position:
                self.move_randomly()
        
        return recorded_successfully
            
    def collect_demonstrations(self, enable_pre_control=None):
        amount = self.options.n_repetitions
        
        if not enable_pre_control:
            enable_pre_control = self.options.enable_pre_control
            
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
                self.logger.exit_logging(self.options, save=False)
                
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
        data = {"unnorm_key": self.options.unnorm_key}
        
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
            img_key = "image" if self.options.mode == "openvla" else "primary_image"
            data[img_key] = self.to_base64(img_primary)
        if cam_wrist:
            img_wrist = cam_wrist.logs[-1].copy()
            img_wrist = self.resize_image(img_wrist, size=128)
            img_wrist = cv2.cvtColor(img_wrist, cv2.COLOR_BGR2RGB)
            if not self.options.mode == "openvla":
                data['wrist_image'] = self.to_base64(img_wrist)
        return data
                
    def run_from_server(self, save=True, evaluating=True, remaining=None):            
        while not self.options.instruction:
            instruction = input(f"Enter instruction (or keep empty to ({instruction}): ")
            self.options.instruction = instruction
            
        instruction = self.options.instruction
        key_pressed = False
        
        def clear_input():
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        
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
        
        if self.options.mode == "openvla":
            self.move_randomly()
        
        begin = None
        clear_input()
        while begin != '':
            begin = input("Press enter to start the trial: ")
        
            
        self.logger.enter_logging()
        for camera in self.logger.cameras:
            while not camera.logs:
                time.sleep(0.1)
                
        print(f"During operation, press \033[92m's'\033[0m for success, \033[91m'f'\033[0m for failure, 'r' for reset")
           
        
        start = time.time()
        while True:
            time_elapsed = round(time.time() - start, 2)
            print(f"\033[KPerforming command ({instruction}) for {self.options.max_seconds} seconds. {round(self.options.max_seconds - time_elapsed,2)}s remaining. {remaining} trials remaining.", end="\r")
            if (time_elapsed) > self.options.max_seconds:
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
                action = requests.post(self.options.ip, json=state).json()
                
                logged_action = action.copy() if self.options.mode == "openvla" else action[0].copy()
                action = action[0] if self.options.mode == "octo" and self.options.execution_horizon == 1 else action
                
                self.logger.log(logged_action, inference=True)
                self.env.step(action)
            except Exception as e:
                print(f"Error in server communication: {e}")
                break
    
        self.env.stop_controller()
        listener.stop()
        self.logger.exit_logging(self.options, save=False, inference=True)
        
        if not key_pressed and evaluating:
            print("\n")
            print(f"Enter \033[92m's'\033[0m for success, \033[91m'f'\033[0m for failure, 'r' for reset: ")
            while key_pressed not in ['s', 'f', 'r']:
                key_pressed = input()
                
        color = "\033[91m" if key_pressed == 'f' else "\033[92m" if key_pressed == 's' else "\033[0m"
        time.sleep(0.5)
        
        clear_input()
        print(f"Input 'enter' to confirm {color}'{key_pressed}'\033[0m or 'r' to reset: ")
        confirmation = None
        while confirmation not in ['', 'r']:
            confirmation = input()
        
        if key_pressed == 'r' or confirmation == 'r':
            print("\n")
            return 'reset'
        
        self.logger.exit_logging(self.options, save=save, inference=True, 
                                 success_or_failure=(key_pressed == 's'), total_time=time_elapsed)
        
        
    def continually_run_from_server(self, instruction=None):
        if instruction:
            self.options.instruction = instruction
        
        i = self.options.n_repetitions 
        
        while i:
            result = self.run_from_server(save=self.options.log, remaining=i), 
            if result == 'reset':
                continue
            i -= 1