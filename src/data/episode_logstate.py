import numpy as np
from dataclasses import dataclass
import pickle
import cv2
import matplotlib.pyplot as plt


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

    franka_resampled_indices: np.ndarray = None
    gripper_resampled_indices: np.ndarray = None
    primarycam_resampled_indices: np.ndarray = None
    wristcam_resampled_indices: np.ndarray = None

    task_description: str = ""

    aligned: bool = False
    filtered_nearzero_velocity: bool = False

    def to_numpy(self, path):
        np.savez(path, **self.__dict__)

    @classmethod
    def from_numpy(cls, path):
        data = np.load(path, allow_pickle=True)
        return cls(**{key: data[key].item() if (key == 'task_description')
                      else data[key] for key in data})

    @classmethod
    def from_pickle(cls, path) -> 'EpisodeLogState':
        with open(path, 'rb') as f:
            return EpisodeLogState(**pickle.load(f).__dict__)

    def assure_equal_lengths(self):
        assert (len(self.franka_t) == len(self.franka_q) == len(self.franka_dq) == len(self.franka_pose) ==
                len(self.gripper_t) == len(self.gripper_status) == len(self.camera_frame_t) ==
                len(self.action) == len(self.wrist_frame_t))

    @staticmethod
    def find_nearest_timestamps(base_timestamps, query_timestamps):
        """
        Note: this may find an entry that violates causality, but that is still nearest in time.
        Since we wish to align cameras, and cameras/actions, this should be ok, since a few tens of ms of difference should
        look reasonably similar in both image and action space.
        """
        base = np.array(base_timestamps)
        query = np.array(query_timestamps)

        # Find nearest indices using array operations
        nearest_indices = []
        for q in query:
            # Calculate absolute differences
            differences = np.abs(base - q)
            # Find index of minimum difference
            nearest_idx = np.argmin(differences)
            nearest_indices.append(nearest_idx)

        return nearest_indices

    def align_logs_with_resampling(self):
        franka_t = np.squeeze((self.franka_t - self.franka_t[0]) / 1e3)
        gripper_t = (self.gripper_t - self.gripper_t[0]) / 1e9
        wrist_frame_t = (self.wrist_frame_t - self.gripper_t[0]) / 1e9
        camera_frame_t = (self.camera_frame_t - self.gripper_t[0]) / 1e9

        franka_resampled_indices = self.find_nearest_timestamps(franka_t, gripper_t)
        gripper_resampled_indices = self.find_nearest_timestamps(gripper_t, gripper_t)
        primarycam_resampled_indices = self.find_nearest_timestamps(camera_frame_t, gripper_t)
        wristcam_resampled_indices = self.find_nearest_timestamps(wrist_frame_t, gripper_t)

        self.franka_q = self.franka_q[franka_resampled_indices]
        self.franka_dq = self.franka_dq[franka_resampled_indices]
        self.franka_pose = self.franka_pose[franka_resampled_indices]
        self.gripper_status = self.gripper_status[gripper_resampled_indices]
        self.action = self.action[gripper_resampled_indices, ::]
        self.franka_t = franka_t[franka_resampled_indices]
        self.gripper_t = gripper_t[gripper_resampled_indices]
        self.wrist_frame_t = wrist_frame_t[wristcam_resampled_indices]
        self.camera_frame_t = camera_frame_t[primarycam_resampled_indices]

        self.assure_equal_lengths()
        self.aligned = True

        self.franka_resampled_indices = franka_resampled_indices
        self.gripper_resampled_indices = gripper_resampled_indices
        self.primarycam_resampled_indices = primarycam_resampled_indices
        self.wristcam_resampled_indices = wristcam_resampled_indices

        return franka_resampled_indices, gripper_resampled_indices, primarycam_resampled_indices, wristcam_resampled_indices

    def remove_near_zero_velocity_frames(self):
        self.assure_equal_lengths()

        gripper_status_lag1 = self.gripper_status[1:]
        gripper_status_lag1 = np.append(gripper_status_lag1, gripper_status_lag1[-1])
        gripper_status_diff = np.abs(gripper_status_lag1 - self.gripper_status)

        gripper_status_diff_extended = np.zeros_like(gripper_status_diff)
        for i in range(len(gripper_status_diff)):
            if gripper_status_diff[i] == 1:
                gripper_status_diff_extended[i:i+3] = 1

        def has_nearzero_velocity(x): return np.sum(np.abs(self.franka_dq[x])) + gripper_status_diff_extended[x] < 0.02
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
        assert (self.aligned and self.task_description)

        data = []
        for i in range(len(self.franka_t)-1):
            data.append({'franka_t': self.franka_t[i], 'franka_q': self.franka_q[i], 'franka_dq': self.franka_dq[i],
                         'franka_pose': self.franka_pose[i], 'gripper_t': self.gripper_t[i], 'gripper_status': self.gripper_status[i],
                         'action': self.action[i], 'camera_frame_t': self.camera_frame_t[i], 'wrist_frame_t': self.wrist_frame_t[i],
                         'task_description': self.task_description})

        return data


if __name__ == '__main__':
    # can use this to preview alignment of camera frames with gripper status
    # for example to check if the cameras align correctly with eachother and the gripper open/close
    episode_name = 'datasets/raw_data/stack_red_blue_100/episode_1.npz'
    primary_camera = 'datasets/raw_data/stack_red_blue_100/primary_episode_1.mp4'
    wrist_camera = 'datasets/raw_data/stack_red_blue_100/wrist_episode_1.mp4'
    episode = EpisodeLogState.from_numpy(episode_name)
    franka_resampled, gripper_resampled, camera_resampled, wrist_resampled = episode.align_logs_with_resampling()

    # load the video files and make them lists
    cap = cv2.VideoCapture(primary_camera)
    primary_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        primary_frames.append(frame)

    cap = cv2.VideoCapture(wrist_camera)
    wrist_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        wrist_frames.append(frame)

    # show the frames from the camera where episode.gripper_status is 1
    # adjusted for resampled indexes
    for i in range(len(episode.gripper_status)):
        if episode.gripper_status[i] == 1:
            primary_frame = camera_resampled[i]
            wrist_frame = wrist_resampled[i]

            print(primary_frame, wrist_frame)

            # show both images with plt as subplots
            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(10, 10)
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[0].imshow(primary_frames[primary_frame])
            ax[1].imshow(wrist_frames[wrist_frame])
            plt.show()
