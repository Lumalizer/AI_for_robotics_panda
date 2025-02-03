from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import os
from episode_logstate import EpisodeLogState
import cv2
import numpy as np
import tqdm


features = {
    "action": {
        "dtype": "float32",
        "shape": (7, ),
        "names": ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "d_gripper"]
    },
    "observation.image_primary": {
        "dtype": "image",
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"]
    },
    "observation.image_wrist": {
        "dtype": "image",
        "shape": (3, 480, 640),
        "names": ["channels", "height", "width"]
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (11, ),
        "names": ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "fk1", "fk2", "fk3", "gripper"]
    }
}


def get_mp4_frames(mp4_path, resampled_indices):
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # make sure to load as RGB, so we do not train on BGR images
        frames.append(frame)

    cap.release()

    frames = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in resampled_indices]
    # make sure images are channel first by swapping the axes
    return np.array(frames).swapaxes(1, 3).swapaxes(2, 3)


def process_episode(episode_path: str, dataset: LeRobotDataset, fps: int):
    try:
        ep = EpisodeLogState.from_numpy(episode_path)
        ep.align_logs_with_resampling(resample_hertz=fps)
        # data.remove_near_zero_velocity_frames()
        data = ep.get_episode_data()
    except Exception as e:
        print(f"Error in processing episode {episode_path}: {e}")
        return None

    primary_mp4_path = episode_path.replace('.npz', '.mp4').replace('episode_', 'primary_episode_')
    wrist_mp4_path = episode_path.replace('.npz', '.mp4').replace('episode_', 'wrist_episode_')

    primary_frames = get_mp4_frames(
        primary_mp4_path, resampled_indices=ep.primarycam_resampled_indices)
    wrist_frames = get_mp4_frames(
        wrist_mp4_path, resampled_indices=ep.wristcam_resampled_indices)

    for i in range(len(data)):
        step = data[i]
        pos = step['franka_pose'][:3, 3]
        gripper_state = np.expand_dims(step['gripper_status'], axis=0)
        state = np.concatenate([step['franka_q'], pos, gripper_state]).astype(np.float32)
        action = step['action'].astype(np.float32)

        dataset.add_frame({
            "action": action,
            "observation.image_primary": primary_frames[i],
            "observation.image_wrist": wrist_frames[i],
            "observation.state": state
        })

    dataset.save_episode(task=step['task_description'])

    return True


def create_lerobot_dataset(dataset_name: str, fps: int, features: dict = features):
    datasets_root = "../datasets"
    used_dataset_path = f"{datasets_root}/raw_data/{dataset_name}"
    numpy_episodes = [p for p in os.listdir(used_dataset_path) if p.endswith(".npz")]

    dataset = LeRobotDataset.create(
        repo_id=f"airnet/{dataset_name}",
        root=f"{datasets_root}/lerobot_datasets/{dataset_name}",
        fps=fps,
        features=features,
    )

    for episode in tqdm.tqdm(numpy_episodes, desc="Processing episodes", colour="green"):
        process_episode(f"{used_dataset_path}/{episode}", dataset, fps)

    dataset.consolidate()
    return dataset


if __name__ == "__main__":
    dataset = create_lerobot_dataset("pick_up_red_100", fps=15)
