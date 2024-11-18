import numpy as np
import pickle
from dataclasses import dataclass


@dataclass
class EpisodeLogState:
    franka_q: np.ndarray
    franka_dq: np.ndarray
    franka_pose: np.ndarray
    gripper_status: np.ndarray
    action: np.ndarray
    franka_t: np.ndarray
    gripper_t: np.ndarray
    wrist_frame_t: np.ndarray
    camera_frame_t: np.ndarray

    task_description: str = ""

    aligned: bool = False
    filtered_nearzero_velocity: bool = False

    @classmethod
    def from_pickle(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def assure_equal_lengths(self):
        assert(len(self.franka_t) == len(self.franka_q) == len(self.franka_dq) == len(self.franka_pose) ==
               len(self.gripper_t) == len(self.gripper_status) == len(self.camera_frame_t) ==
               len(self.action) == len(self.wrist_frame_t))

    def save_raw_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def align_logs_with_resampling(self):
        franka_t = np.squeeze((self.franka_t - self.franka_t[0]) / 1e3)
        gripper_t = (self.gripper_t - self.gripper_t[0]) / 1e9
        wrist_frame_t = (self.wrist_frame_t - self.gripper_t[0]) / 1e9
        camera_frame_t = (self.camera_frame_t - self.gripper_t[0]) / 1e9

        franka_resampled_indices = np.searchsorted(franka_t, camera_frame_t)
        gripper_resampled_indices = np.searchsorted(gripper_t, camera_frame_t)
        wrist_resampled_indices = np.searchsorted(wrist_frame_t, camera_frame_t)

        if franka_resampled_indices[-1] == len(franka_t):
            franka_resampled_indices[-1] = len(franka_t) - 1
        if gripper_resampled_indices[-1] == len(gripper_t):
            gripper_resampled_indices[-1] = len(gripper_t) - 1
        if wrist_resampled_indices[-1] == len(wrist_frame_t):
            wrist_resampled_indices[-1] = len(wrist_frame_t) - 1

        self.franka_q = self.franka_q[franka_resampled_indices]
        self.franka_dq = self.franka_dq[franka_resampled_indices]
        self.franka_pose = self.franka_pose[franka_resampled_indices]
        self.gripper_status = self.gripper_status[gripper_resampled_indices]
        self.action = self.action[gripper_resampled_indices, ::]
        self.franka_t = franka_t[franka_resampled_indices]
        self.gripper_t = gripper_t[gripper_resampled_indices]
        self.wrist_frame_t = wrist_frame_t[wrist_resampled_indices]
        self.camera_frame_t = camera_frame_t

        self.assure_equal_lengths()
        self.aligned = True

    def remove_near_zero_velocity_frames(self):
        self.assure_equal_lengths()

        gripper_status_lag1 = self.gripper_status[1:]
        gripper_status_lag1 = np.append(gripper_status_lag1, gripper_status_lag1[-1])
        gripper_status_diff = np.abs(gripper_status_lag1 - self.gripper_status)

        gripper_status_diff_extended = np.zeros_like(gripper_status_diff)
        for i in range(len(gripper_status_diff)):
            if gripper_status_diff[i] == 1:
                gripper_status_diff_extended[i:i+3] = 1

        has_nearzero_velocity = lambda x: np.sum(np.abs(self.franka_dq[x])) + gripper_status_diff_extended[x] < 0.02
        indexes_to_keep = [i for i in range(len(self.franka_dq)) if not has_nearzero_velocity(i)]

        self.franka_t = self.franka_t[indexes_to_keep]
        self.franka_q = self.franka_q[indexes_to_keep]
        self.franka_dq = self.franka_dq[indexes_to_keep]
        self.franka_pose = self.franka_pose[indexes_to_keep]
        self.gripper_t = self.gripper_t[indexes_to_keep]
        self.gripper_status = self.gripper_status[indexes_to_keep]
        self.action = self.action[indexes_to_keep]
        self.camera_frame_t = self.camera_frame_t[indexes_to_keep]
        self.wrist_frame_t = self.wrist_frame_t[indexes_to_keep]

        self.assure_equal_lengths()
        self.filtered_nearzero_velocity = True

    def get_episode_data(self):
        self.assure_equal_lengths()
        assert(self.aligned and self.filtered_nearzero_velocity and self.task_description)

        data = []
        for i in range(len(self.franka_t)-1):
            data.append({'franka_t':self.franka_t[i], 'franka_q':self.franka_q[i], 'franka_dq':self.franka_dq[i],
                         'franka_pose':self.franka_pose[i], 'gripper_t':self.gripper_t[i], 'gripper_status':self.gripper_status[i],
                         'action':self.action[i], 'camera_frame_t':self.camera_frame_t[i], 'wrist_frame_t':self.wrist_frame_t[i],
                         'task_description':self.task_description})

        return data