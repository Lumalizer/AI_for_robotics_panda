import time
import os
import cv2
import numpy as np
import panda_py
from in_out.camera import Camera
import threading

class Logger:
    def __init__(self, fc: 'FrankaController', task_description_required=True, fps=15.0) -> None:
        self.fc = fc
        self.camera = Camera()
        self.fps = fps
        
        self.task_description_required = task_description_required
        self.clear_logs()
        
    def clear_logs(self):
        self._logs = {'gripper': [0], 'time': [time.time_ns()], 'action': [np.zeros(7)]}
        self._camera_logs = []
        self._camera_time = []
        
    def enter_logging(self):
        self.fc.is_recording.set()
        # logs directly from libfranka
        seconds_to_log = (self.fc.max_runtime if self.fc.max_runtime > 0 else 600)
        self.fc.panda.enable_logging(buffer_size=seconds_to_log * 1000)

        # our own logs (gripper / camera)
        self.clear_logs()

        self.camera.start()
        
        if not hasattr(self, 'cam_thread'):
            self._start_camera_thread()
            

    def camera_thread_fn(self):
        while True:
            if self.fc.is_recording.is_set():
                self.camera.start()
            while self.fc.is_recording.is_set():
                try:
                    self._camera_logs.append(self.camera.get_frame())
                    self._camera_time.append(time.time_ns())
                except Exception as e:
                    print(f"Error in camera thread: {e}")
                    self.camera.stop()
                    raise e
            else:
                self.camera.stop()
                
            time.sleep(0.01)

    def _start_camera_thread(self):
        try:
            self.cam_thread = threading.Thread(target=self.camera_thread_fn, daemon=True)
            self.cam_thread.start()
        except Exception as e:
            print(f"Camera not connected.")
            raise e
        
    def exit_logging(self, save=True):
        self.fc.is_recording.clear()
        self.fc.panda.disable_logging()
        
        if save:
            logs = self.get_resampled_logs()
            
            if self.task_description_required:
                task_desc = input("Enter task description such as 'pick up the red cube',  or press enter to leave blank: ")
                
                for log in logs:
                    log['task_description'] = task_desc
            
            # date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            dataset_path = os.path.join("../datasets", self.fc.dataset_name)
            os.makedirs(dataset_path, exist_ok=True)
            
            amount_episodes = len(os.listdir(dataset_path)) // 2
            episode_path = os.path.join(dataset_path, f'episode_{amount_episodes+1}.npy')
            mp4_path = os.path.join(dataset_path, f'episode_{amount_episodes+1}.mp4')
            
            np.save(episode_path, logs)
            self.write_mp4(mp4_path)
            
            print(f'Trajectory saved to {dataset_path}. Camera frames: {len(self._camera_logs)}, Camera fps (assuming 100 gripper logs/s): {len(self._camera_logs) / (len(self._logs["gripper"]) / 100)} Gripper frames: {len(self._logs["gripper"])} Gripper frames closed: {sum(self._logs["gripper"])}\n')
            print(f'Episode saved to {episode_path}')
            
    def write_mp4(self, camera_path): # might be distorting the colors of the frames while creating the mp4
        frame_height, frame_width = self._camera_logs[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter(camera_path, fourcc, self.fps, (frame_width, frame_height))

        for frame in self._camera_logs:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
            
        out.release()
        
    def get_resampled_logs(self):
        logs = self.fc.panda.get_log()
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

        camera_frame_t = np.array(self._camera_time)
        camera_frame_t = (camera_frame_t - self._logs['time'][0]) / 1e9
        
        # now we re-sample gripper (gripper_status / gripper_t) and franka_* to camera_frame_t (i.e., 30 Hz) by looking for the indices whose timestamp is closest to the camera_frame_t
        franka_resampled_indices = np.searchsorted(franka_t, camera_frame_t)
        gripper_resampled_indices = np.searchsorted(gripper_t, camera_frame_t)

        if franka_resampled_indices[-1] == len(franka_t):
            franka_resampled_indices[-1] = len(franka_t) - 1
        if gripper_resampled_indices[-1] == len(gripper_t):
            gripper_resampled_indices[-1] = len(gripper_t) - 1

        #print(len(franka_t), len(franka_resampled_indices), franka_resampled_indices[-1], len(action))

        franka_t = franka_t[franka_resampled_indices]
        franka_q = franka_q[franka_resampled_indices]
        franka_dq = franka_dq[franka_resampled_indices]
        franka_pose = franka_pose[franka_resampled_indices]

        gripper_t = gripper_t[gripper_resampled_indices]
        gripper_status = gripper_status[gripper_resampled_indices]

        # action is relative, so we should add actions together in between resampled indices
        action = np.cumsum(action, axis=0)
        gripper_resampled_indices = gripper_resampled_indices.tolist()
        action = np.diff(action[ gripper_resampled_indices+[len(action)-1] ],axis=0)

        print(len(franka_t), len(gripper_status), len(action))


        # Now we have all the data we need, timestamp-aligned and sub-sampled at the same frame rate as the camera (ideally, 30Hz, if data collected through
        # usb-3 port)
        
        assert(len(franka_t) == len(franka_q) == len(franka_dq) == len(franka_pose) == len(gripper_t) == len(gripper_status) == len(camera_frame_t) == len(action))
        
        data = []
                
        for i in range(len(franka_t)-1):
            data.append({'franka_t':franka_t[i], 'franka_q':franka_q[i], 'franka_dq':franka_dq[i], 'franka_pose':franka_pose[i], 'gripper_t':gripper_t[i], 'gripper_status':gripper_status[i], 'action':action[i], 'camera_frame_t':camera_frame_t[i]})
        
        return data
    
    def log_gripper(self):
        # TODO : gripper.read_once() is blocking, check if solution is OK
        # self._logs['gripper'].append(self.fc.gripper.read_once().is_grasped)
        self._logs['gripper'].append(self.fc.is_gripping)
        self._logs['time'].append(time.time_ns())

    def log_action(self, action):
        self._logs['action'].append(action)
        
    def get_camera_frame_resized(self, size=(256, 256)):
        image_primary = self._camera_logs[-1]
        image_primary = cv2.resize(image_primary, size)
        return image_primary
        

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


