import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation
from panda_py import constants
constants.JOINT_LIMITS_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
constants.JOINT_LIMITS_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
import panda_py
from panda_py import libfranka
from panda_py import controllers
from in_out.spacemousecontroller import SpaceMouseController, SpaceMouseState
from in_out.logger import Logger
from controller.franka_runner import FrankaRunner
import cv2

class FrankaController:
    def __init__(self, 
                 logger:Logger=None, 
                 conversion_factor=0.003, 
                 angle_conversion_factor=0.4, 
                 mouse_axes_conversion=SpaceMouseState(1, 1, 1, 1, 1, 1), 
                 dataset_name="test_franka_ds",
                 max_runtime=-1,
                 runner: FrankaRunner=None):
        
        self.panda = panda_py.Panda("172.16.0.2")
        self.gripper = libfranka.Gripper("172.16.0.2")
        self.runner = runner
        
        self.dataset_name = dataset_name
        self.max_runtime = max_runtime
        
        try:
            self.spacemouse_controller = SpaceMouseController(self.button_callback, conversion_factor, angle_conversion_factor, mouse_axes_conversion)
        except Exception as e:
            self.spacemouse_controller = None
            print("SpaceMouse not connected.")
        
        self.is_gripping = False
        self.is_recording = threading.Event()
        
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
        
    def get_pose_components(self):
        pose = self.panda.get_pose()
        # what is the difference to pose = panda_py.fk(q) ?
        pos = pose[:3, 3]
        angles = Rotation.from_matrix(pose[:3, :3]).as_quat()
        return pos, angles
    
    @staticmethod
    def add_rot_to_quat(angles, new_rot, degrees=True):
        rot = Rotation.from_quat(angles)
        delta_rot = Rotation.from_euler('zyx', new_rot, degrees=degrees)
        rot = rot * delta_rot
        return rot.as_quat()
    
    def get_current_state_for_inference(self) -> tuple[dict, np.ndarray, np.ndarray]:
        state = self.panda.get_state()
        pos, angles = self.get_pose_components()
        
        # use fk on q
        # add dq and xyz
        
        # gripper_status = np.array([self.gripper.read_once().is_grasped])
        # TODO fix gripper blocking
        gripper_status = np.array([0])
        mask = np.array([1])
        
        img = self.logger.get_camera_frame_resized()
        img = np.expand_dims(img, axis=0)
        
        state = np.concatenate([state.q, pos, gripper_status])
        state = np.expand_dims(state, axis=0)
        return {'proprio': state, 'image_primary': img, 'timestep_pad_mask': mask}, pos, angles      

    def enable_spacemouse_control(self, log=True):
        self.is_gripping = self.gripper.read_once().is_grasped
        pos, angles = self.get_pose_components()

        print(f"Starting SpaceMouse control for {self.max_runtime} seconds...")

        self.panda.start_controller(self.ctrl)
        log and self.logger.enter_logging()
        
        with self.panda.create_context(frequency=1e2, max_runtime=self.max_runtime) as ctx:
            while ctx.ok() and self.is_recording.is_set():
                self.logger.log_gripper()

                mouse = self.spacemouse_controller.read()
    
                delta_pos = np.array([-mouse.y, mouse.x, mouse.z])
                delta_rot = np.array([mouse.yaw, mouse.pitch, -mouse.roll])

                pos += delta_pos
                angles = self.add_rot_to_quat(angles, delta_rot, degrees=True)

                # TODO: log actions
                self.logger.log_action( np.array([*delta_pos, *delta_rot, self.is_gripping]) )

                self.ctrl.set_control(pos, angles)
        
        self.panda.stop_controller()
        log and self.logger.exit_logging()

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
                
    def run_with_model(self):
        if self.runner is None:
            raise ValueError("Runner not set.")
        
        """
        # while True:
            
        self.reset_robot_position()
        
        # goal_instruction = ""        
        # print("Current instruction: ", goal_instruction)
        # if click.confirm("Take a new instruction?", default=True):
        #     text = input("Instruction?")
        # goal_instruction = text
            
        text="pick up the blue cube"    
        task = self.runner.model.create_tasks(texts=[text])
        
        self.logger.enter_logging()
        
        while not self.logger._camera_logs:
            time.sleep(0.1)
        
        self.panda.start_controller(self.ctrl)
        
        with self.panda.create_context(frequency=1, max_runtime=-1) as ctx:
            while ctx.ok():
                obs, pos, angles = self.get_current_state_for_inference() # q, pos, gripper input for inference                
                delta_xyz, delta_rot, grip = self.runner.infer(obs, task)
                
                # TODO : inference above keeps interrupting control loop --> new thread?
                # it works fine with the dummy values below (comment out infer)
                # delta_xyz = np.array([.01,.01,.01])
                # delta_rot = np.array([.01,.01,.01])

                absolute_xyz = pos + delta_xyz
                absolute_xyz = np.expand_dims(absolute_xyz, axis=1)

                absolute_orientation = self.add_rot_to_quat(angles, delta_rot, degrees=False)
                absolute_orientation = np.expand_dims(absolute_orientation, axis=1)
                
                self.ctrl.set_control(absolute_xyz, absolute_orientation)
                   
        self.panda.stop_controller()
        self.logger.exit_logging(save=False)
        """


if __name__ == "__main__":    
    fc = FrankaController(dataset_name="test_franka_ds")
    fc.collect_demonstrations()
    
    