import time
import os
import cv2
import numpy as np
import panda_py
from in_out.camera.RealSenseCamera import RealSenseCamera
from in_out.camera.LogitechCamera import LogitechCamera
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, fc: 'FrankaController', task_description_required=True, fps=30.0) -> None:
        self.fc = fc
        self.camera = LogitechCamera(is_recording=self.fc.is_recording)
        self.fps = fps
        
        self.previous_task_desc = ""
        
        self.task_description_required = task_description_required
        self.clear_logs()

        self.camera.start()
        
    def clear_logs(self):
        self._logs = {'gripper': [0], 'time': [time.time_ns()], 'action': [np.zeros(7)]}
        self.camera.clear_logs()
        
    def enter_logging(self):
        self.fc.is_recording.set()
        # logs directly from libfranka
        seconds_to_log = (self.fc.max_runtime if self.fc.max_runtime > 0 else 600)
        self.fc.env.enable_logging(buffer_size=seconds_to_log * 1000)

        # our own logs (gripper / camera)
        self.clear_logs()
        
        if not hasattr(self.camera, 'cam_thread'):
            self.camera.start_camera_thread()
        
    def exit_logging(self, save=True):
        self.fc.is_recording.clear()
        self.fc.env.disable_logging()
        
        if save:
            logs = self.get_resampled_logs()
            
            if self.task_description_required:
                task_desc = input(f"""Enter task description such as 'pick up the red cube'
                                  or press ENTER to re-use the previous task description: ({self.previous_task_desc})
                                  or type 'skip' to skip saving this trajectory: """).strip()
                if task_desc == 'skip':
                    return False
                elif not task_desc:
                    task_desc = self.previous_task_desc
                
                self.previous_task_desc = task_desc
                for log in logs:
                    log['task_description'] = task_desc
            
            dataset_path = os.path.join("datasets", self.fc.dataset_name)
            os.makedirs(dataset_path, exist_ok=True)
            
            # fill in the gaps in the episode numbers if needed
            files = os.listdir(dataset_path)
            existing_ep_nums = [int(f.split("episode_")[1].replace(".npy", ""))for f in files if f.endswith('.npy')]
            
            if not existing_ep_nums:
                ep_num = 1
            else:
                for i in range(1, len(existing_ep_nums)+2):
                    if i not in existing_ep_nums:
                        ep_num = i
                        break
            
            episode_path = os.path.join(dataset_path, f'episode_{ep_num}.npy')
            mp4_path = os.path.join(dataset_path, f'episode_{ep_num}.mp4')
            
            np.save(episode_path, logs)
            self.write_mp4(mp4_path)
            
            # keep a text file with some metadata
            metadata_exists = os.path.exists(os.path.join(dataset_path, 'data.csv'))
            with open(os.path.join(dataset_path, 'data.csv'), 'a') as f:
                if not metadata_exists:
                    f.write("episode,task_description,n_frames\n")
                f.write(f"{ep_num},{task_desc},{len(logs)}\n")
            
            print(f'Trajectory saved to {dataset_path}. Camera frames: {len(self.camera.logs)}, Camera fps (assuming 100 gripper logs/s): {len(self.camera.logs) / (len(self._logs["gripper"]) / 100)} Gripper frames: {len(self._logs["gripper"])} Gripper frames closed: {sum(self._logs["gripper"])}\n')
            print(f'Episode saved to {episode_path}')
            
            return True
            
    def write_mp4(self, camera_path): # might be distorting the colors of the frames while creating the mp4
        frame_height, frame_width = self.camera.logs[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter(camera_path, fourcc, self.fps, (frame_width, frame_height))

        for frame in self.camera.logs:
            out.write(frame)
            
        out.release()
        
    def get_resampled_logs(self):
        logs = self.fc.env.get_log()
        franka_q = np.array(logs['q'])
        franka_dq = np.array(logs['dq'])
        franka_pose = np.array([panda_py.fk(qq) for qq in franka_q])
        franka_t = np.array(logs['time'])
        gripper_status = np.array(self._logs['gripper'])
        action = np.array(self._logs['action'])  # action executed at time t, so this is already the target!
        
        franka_t = (franka_t-franka_t[0]) / 1e3
        franka_t = np.squeeze(franka_t)

        gripper_t = np.array(self._logs['time'])
        gripper_t = (gripper_t - gripper_t[0]) / 1e9

        camera_frame_t = np.array(self.camera.time)
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
        action = action[gripper_resampled_indices, ::]        
        
        # remove near-zero velocity frames
        has_nearzero_velocity = lambda x: np.sum(np.abs(x)) < 0.02
        indexes_to_keep = [i for i in range(len(franka_dq)) if not has_nearzero_velocity(franka_dq[i])]
        
        franka_t = franka_t[indexes_to_keep]
        franka_q = franka_q[indexes_to_keep]
        franka_dq = franka_dq[indexes_to_keep]
        franka_pose = franka_pose[indexes_to_keep]
        gripper_t = gripper_t[indexes_to_keep]
        gripper_status = gripper_status[indexes_to_keep]
        action = action[indexes_to_keep]
        camera_frame_t = camera_frame_t[indexes_to_keep]

        # Now we have all the data we need, timestamp-aligned and sub-sampled at the same frame rate as the camera (ideally, 30Hz, if data collected through
        # usb-3 port)
        
        assert(len(franka_t) == len(franka_q) == len(franka_dq) == len(franka_pose) == len(gripper_t) == len(gripper_status) == len(camera_frame_t) == len(action))
        
        data = []
                
        for i in range(len(franka_t)-1):
            data.append({'franka_t':franka_t[i], 'franka_q':franka_q[i], 'franka_dq':franka_dq[i], 
                         'franka_pose':franka_pose[i], 'gripper_t':gripper_t[i], 'gripper_status':gripper_status[i], 
                         'action':action[i], 'camera_frame_t':camera_frame_t[i]})
        
        return data

    def plot_velocity_sums(self, franka_dq):
        vels = []
        for i in range(len(franka_dq)):
            # add sum of velocities
            sum_vel = np.sum(np.abs(franka_dq[i]))
            vels.append(sum_vel)

        plt.plot(vels)
        plt.show()
    
    def log_gripper(self):
        # TODO : gripper.read_once() is blocking, check if solution is OK
        # self._logs['gripper'].append(self.fc.gripper.read_once().is_grasped)
        self._logs['gripper'].append(self.fc.is_gripping)
        self._logs['time'].append(time.time_ns())

    def log_action(self, action):
        self._logs['action'].append(action)
        

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


