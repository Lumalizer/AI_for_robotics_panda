from franka_spacemouse import FrankaController
import time
import datetime
import os
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
                task_desc = input("Enter task description such as 'pick up the red cube',  or press enter to leave blank: ")
                
                for log in logs:
                    log['task_description'] = task_desc
            
            ## save the raw data
            date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            raw_path = os.path.join("logs", self.fc.dataset_name, "trajectory_"+str(date))
            os.makedirs(raw_path, exist_ok=True)
        
            resize_img = [cv2.resize(frame, (256, 256)) for frame in self._camera_logs]
            # save each frame in the trajectory.npy as a dictionary with a key image
            for i, frame in enumerate(resize_img):
                # np.save(os.path.join(path, f'image_{i}.npy'), frame)
                if i < len(logs):
                    logs[i]['image'] = frame
            
            np.save(os.path.join(raw_path, 'trajectory.npy'), logs)
            self.write_mp4(raw_path)
            
            ## save data in npy for easy dataset build
            dataset_path = os.path.join("datasets", self.fc.dataset_name)
            os.makedirs(dataset_path, exist_ok=True)
            amount_episodes = len(os.listdir(dataset_path))
            episode_path = os.path.join(dataset_path, f'episode_{amount_episodes+1}.npy')
            np.save(episode_path, logs)
            
            print(f'Trajectory saved to {raw_path}. Camera frames: {len(self._camera_logs)}, Camera fps (assuming 100 gripper logs/s): {len(self._camera_logs) / (len(self._logs["gripper"]) / 100)} Gripper frames: {len(self._logs["gripper"])} Gripper frames closed: {sum(self._logs["gripper"])}\n')
            print(f'Episode saved to {dataset_path}', f'episode_{amount_episodes+1}')
                  
    def write_mp4(self, camera_path): # might be distorting the colors of the frames
        frame_height, frame_width = self._camera_logs[0].shape[:2]
        video_path = os.path.join(camera_path, "video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (frame_width, frame_height))

        for frame in self._camera_logs:
            out.write(frame)
            
        out.release()
        
    def process_log(self):
        logs = self.fc.panda.get_log()

        franka_q = np.array(logs['q'])
        franka_dq = np.array(logs['dq'])
        
        franka_t = np.array(logs['time'])
        franka_t = (franka_t-franka_t[0]) / 1e3
        franka_t = np.squeeze(franka_t)
        franka_pose = np.array([panda_py.fk(qq) for qq in franka_q])

        gripper_status = np.array(self._logs['gripper'])
        gripper_t = np.array(self._logs['time'])
        gripper_t = (gripper_t - gripper_t[0]) / 1e9

        camera_frame_t = np.array(self._camera_time)
        camera_frame_t = (camera_frame_t - self._logs['time'][0]) / 1e9
        
        # now we re-sample gripper (gripper_status / gripper_t) and franka_* to camera_frame_t (i.e., 30 Hz) by looking for the indices whose timestamp is closest to the camera_frame_t
        franka_resampled_indices = np.searchsorted(franka_t, camera_frame_t)
        gripper_resampled_indices = np.searchsorted(gripper_t, camera_frame_t)

        if franka_resampled_indices[-1] == len(franka_t):
            franka_resampled_indices[-1] = len(franka_t) - 1
        if gripper_resampled_indices[-1] == len(gripper_t):
            gripper_resampled_indices[-1] = len(gripper_t) - 1

        franka_t = franka_t[franka_resampled_indices]
        franka_q = franka_q[franka_resampled_indices]
        franka_dq = franka_dq[franka_resampled_indices]
        franka_pose = franka_pose[franka_resampled_indices]

        gripper_t = gripper_t[gripper_resampled_indices]
        gripper_status = gripper_status[gripper_resampled_indices]


        # Now we have all the data we need, timestamp-aligned and sub-sampled at the same frame rate as the camera (ideally, 30Hz, if data collected through
        # usb-3 port)
        
        assert(len(franka_t) == len(franka_q) == len(franka_dq) == len(franka_pose) == len(gripper_t) == len(gripper_status) == len(camera_frame_t))
        
        data = []
                
        for i in range(len(franka_t)-1):
            data.append({'franka_t':franka_t[i], 'franka_q':franka_q[i], 'franka_dq':franka_dq[i], 'franka_pose':franka_pose[i], 'gripper_t':gripper_t[i], 'gripper_status':gripper_status[i], 'camera_frame_t':camera_frame_t[i]})
        
        return data







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


