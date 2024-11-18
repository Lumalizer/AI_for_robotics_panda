import time
import os
import cv2
import numpy as np
import panda_py
from in_out.episode_logstate import EpisodeLogState
from in_out.camera.BaseCamera import BaseCamera
from in_out.camera.LogitechCamera import LogitechCamera
from in_out.camera.RealSenseCamera import RealSenseCamera
import matplotlib.pyplot as plt
import threading

class Logger:
    def __init__(self, fc: 'FrankaController', fps, show_cameras=True) -> None:
        self.fc = fc
        
        self.fps = fps
        self.previous_task_desc = ""
        
        self.cameras: list[BaseCamera] = []
        self.cameras.append(LogitechCamera(name="primary", is_recording=self.fc.is_recording, fps=fps))
        self.cameras.append(RealSenseCamera(name="wrist", is_recording=self.fc.is_recording, fps=fps))        
        
        self.clear_logs()

        for camera in self.cameras:
            camera.start_camera_thread()

        self.show_cameras = show_cameras
        if self.show_cameras:
            self.show_cameras_thread = threading.Thread(target=self.update_cameras_thread, daemon=True)
            self.show_cameras_thread.start()

    def update_cameras_thread(self):
        for camera in self.cameras:
            cv2.namedWindow(camera.name, cv2.WINDOW_AUTOSIZE)
        while True:
            for camera in self.cameras:
                if camera.last_frame is not None:
                    viz = camera.last_frame.copy()
                    viz = cv2.resize(viz, (0, 0), fx=1, fy=1)
                    cv2.imshow(camera.name, viz)
            cv2.waitKey(1)
            
    @property
    def primary_camera(self):
        for camera in self.cameras:
            if camera.name == "primary":
                return camera
            
    @property
    def wrist_camera(self):
        for camera in self.cameras:
            if camera.name == "wrist":
                return camera
        
    def clear_logs(self):
        self._logs = {'gripper': [0], 'time': [time.time_ns()], 'action': [np.zeros(7)]}
        
        for camera in self.cameras:
            camera.clear_logs()
        
    def log(self, action):
        self._logs['action'].append(action)
        self._logs['gripper'].append(self.fc.is_gripping)
        self._logs['time'].append(time.time_ns())
        
    def enter_logging(self):
        self.fc.is_recording.set()
        # logs directly from libfranka
        seconds_to_log = 600
        self.fc.env.enable_logging(buffer_size=seconds_to_log * 1000)

        # our own logs (gripper / camera)
        self.clear_logs()
        
        for camera in self.cameras:
            if not hasattr(camera, 'cam_thread'):
                camera.start_camera_thread()
        
    def exit_logging(self, save=True):
        self.fc.is_recording.clear()
        self.fc.env.disable_logging()
        
        if save:
            logs = self.get_raw_logstate()
            
            
            task_desc = input(f"""Enter task description such as 'pick up the red cube'
                                or press ENTER to re-use the previous task description: ({self.previous_task_desc})
                                or type 'skip' to skip saving this trajectory: """).strip()
            if task_desc == 'skip':
                return False
            elif not task_desc:
                task_desc = self.previous_task_desc
            
            self.previous_task_desc = task_desc
            logs.task_description = task_desc
            
            dataset_path = os.path.join("datasets", "raw_data", self.fc.dataset_name)
            os.makedirs(dataset_path, exist_ok=True)
            
            # fill in the gaps in the episode numbers if needed
            ep_num = self.get_ep_num_to_log(dataset_path)
            
            episode_path = os.path.join(dataset_path, f'episode_{ep_num}.pkl')
            logs.save_raw_pickle(episode_path)
            
            for camera in self.cameras:
                self.write_mp4(camera, dataset_path, ep_num)

            # keep a text file with some metadata
            metadata_exists = os.path.exists(os.path.join(dataset_path, 'data.csv'))
            with open(os.path.join(dataset_path, 'data.csv'), 'a') as f:
                if not metadata_exists:
                    f.write("episode,task_description\n")
                f.write(f"{ep_num},{task_desc}\n")
            
            print(f'Trajectory saved to {episode_path}. \n')
            print(f'Gripper frames: {len(self._logs["gripper"])} \nGripper frames closed: {sum(self._logs["gripper"])}\n\n')
            for camera in self.cameras:
                print(f'{camera.name} camera frames: {len(camera.logs)}, \n{camera.name} camera fps (assuming 100 gripper logs/s): {len(camera.logs) / (len(self._logs["gripper"]) / 100)}' )
            
            return True

    def get_ep_num_to_log(self, dataset_path):
        files = os.listdir(dataset_path)
        existing_ep_nums = [int(f.split("episode_")[1].replace(".npy", ""))for f in files if f.endswith('.npy')]
            
        if not existing_ep_nums:
            return 1
        else:
            for i in range(1, len(existing_ep_nums)+2):
                if i not in existing_ep_nums:
                    return i
        
    def write_mp4(self, camera: BaseCamera, dataset_path: str, ep_num: int):
        camera_path = os.path.join(dataset_path, f'{camera.name}_episode_{ep_num}.mp4')
        frame_height, frame_width = camera.logs[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter(camera_path, fourcc, camera.fps, (frame_width, frame_height))
        
        for frame in camera.logs:
            out.write(frame)
        
        out.release()
    
    def get_raw_logstate(self):
        logs = self.fc.env.get_log()

        logstate = EpisodeLogState(
            franka_q=np.array(logs['q']),
            franka_dq=np.array(logs['dq']),
            franka_pose=np.array([panda_py.fk(qq) for qq in logs['q']]),
            gripper_status=np.array(self._logs['gripper']),
            action=np.array(self._logs['action']),
            franka_t=np.array(logs['time']),
            gripper_t=np.array(self._logs['time']),
            wrist_frame_t=np.array(self.wrist_camera.time),
            camera_frame_t=np.array(self.primary_camera.time)
        )
        
        return logstate

    def plot_velocity_sums(self, franka_dq):
        vels = []
        for i in range(len(franka_dq)):
            # add sum of velocities
            sum_vel = np.sum(np.abs(franka_dq[i]))
            vels.append(sum_vel)

        plt.plot(vels)
        plt.show()
        

if __name__ == "__main__":
    from controller.franka_controller import FrankaController


# e.g., a dataset for diffusion policy may have inputs/state as (img{t}, franka_q{t}, franka_dq{t} or img{t}, franka_pose{t}) and output as (franka_pose {t+1 : }, gripper_status {t+1 : })

# x = franka_pose[:, 0, 3]
# y = franka_pose[:, 1, 3]
# z = franka_pose[:, 2, 3]

# # plot traj in 3d
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z)
# ax.scatter(x,y,z)
# # add limits
# ax.set_xlim([0, 0.9])
# ax.set_ylim([-0.5, 0.5])
# ax.set_zlim([0, 0.8])

# # plt.show()


