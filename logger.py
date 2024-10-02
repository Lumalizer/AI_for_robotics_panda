from franka_spacemouse import FrankaController
import time
import datetime
import os
import pickle
import cv2
import numpy as np
import panda_py

class Logger:
    def __init__(self, fc: FrankaController, task_description_required=True) -> None:
        self.fc = fc
        
        self.task_description_required = task_description_required
        self._camera_logs = []
        self._camera_time = []
        self._logs = {'gripper': [0], 'time': [time.time_ns()]}
        
    def enter_logging(self):
        if self.fc.camera is None:
            self.fc._start_camera_thread()
            
        # logs directly from libfranka
        seconds_to_log = (self.fc.max_runtime if self.fc.max_runtime > 0 else 600)
        self.fc.panda.enable_logging(buffer_size=seconds_to_log * 1000)

        # our own logs (gripper / camera)
        self._logs = {'gripper': [0], 'time': [time.time_ns()]}
        self._camera_logs = []
        self._camera_time = []
        
        self.fc.is_recording.set()
        
    def exit_logging(self, save=True):
        self.fc.is_recording.clear()
        self.fc.panda.disable_logging()
        
        if save:
            logs = self.process_log()
            
            if self.task_description_required:
                logs['task_description'] = input("Enter task description such as 'pick up the red cube',  or press enter to leave blank: ")
            
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = os.path.join("logs", "trajectory_"+str(date))
            camera_path = os.path.join(path, "camera")
            os.makedirs(camera_path, exist_ok=True)

            with open(os.path.join(path, 'trajectory.pkl'), 'wb') as f:
                pickle.dump(logs, f)

            self.write_mp4(camera_path)
            
            print(f'Trajectory saved to {path}. Camera frames: {len(self._camera_logs)}, Camera fps (assuming 100 gripper logs/s): {len(self._camera_logs) / (len(self._logs["gripper"]) / 100)} Gripper frames: {len(self._logs["gripper"])} Gripper frames closed: {sum(self._logs["gripper"])}\n')

    def write_mp4(self, camera_path):
        frame_height, frame_width = self._camera_logs[0].shape[:2]
        video_path = os.path.join(camera_path, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (frame_width, frame_height))

        for frame in self._camera_logs:
            out.write(frame)
            
        out.release()
        
    def process_log(self):
        logs = self.fc.panda.get_log()

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

        data = {'franka_t':t, 'franka_q':np.array(q), 'franka_dq':np.array(dq), 'franka_pose':np.array(poses), 'gripper_t':gripper_time, 'gripper_status':gripper, 'camera_frame_t':cam_time}
        
        return data
