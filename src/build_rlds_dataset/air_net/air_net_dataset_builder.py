from typing import Iterator, Tuple, Any
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import cv2
import pickle
import os
import sys

p = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src/'))
sys.path.append(p)
from data.episode_logstate import EpisodeLogState  # noqa: E402

# for multiprocessing
# we need to add the module folder to python path for multiprocessing
# $env:PYTHONPATH = "$env:PYTHONPATH;D:\Dev\AI_for_robotics_panda\src\build_rlds_dataset"
# p2 = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src/build_rlds_dataset'))
# sys.path.append(p2)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class AirNet(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.24')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
        '1.0.1': 'hover_simple_ds',
        '1.0.2': 'hover_simple_ds_extend',
        '1.0.3': 'hover_blue_logi',
        '1.0.4': 'hover_blue_logi2',
        '1.0.5': 'hover_blue_logi3',
        '1.0.6': 'first_grasp',
        '1.0.7': 'wrist_cam_test',
        '1.0.8': 'octo_with_wrist',
        '1.0.9': 'octo_with_wrist_fixed',
        '1.0.10': 'octo_with_wrist_fixed',
        '1.0.11': 'octo_with_wrist_RAW',
        '1.0.12': 'octo_with_wrist_RAW_diagnostic_close',
        '1.0.13': 'octo_with_wrist_RAW_diagnostic_wide',
        '1.0.14': 'octo_with_wrist_RAW_diagnostic_wide',
        '1.0.15': 'grasp_blue_300',
        '1.0.16': 'grasp_blue300red100_blue_from_close_100_pick_up_blue200_recover_50',
        '1.0.17': 'pickup_blue200_stack_bluered100_redblue100',
        '1.0.18': 'pick_up_blue_200_30hz',
        '1.0.19': 'pick_up_blue_200_15hz',
        '1.0.20': 'pick_up_blue_200',
        '1.0.21': '14datasets_05_12_2024_recover_stack_knockover_pack_pickup_place_unpack',
        '1.0.22': '21datasets_06_12_2024_recover_stack_knockover_pack_pickup_place_unpack',
        '1.0.23': 'ironing_block_25',
        '1.0.24': 'pickup_200blue_100red_stack_100blue_red_100red_blue',
    }
    # make sure the name matches the folder

    RELEASE_NAME = RELEASE_NOTES[str(VERSION)]
    dataset_path = f'../../../datasets/raw_data/{RELEASE_NAME}'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # assert(os.path.exists(self.dataset_path))
        # self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'primary_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(11,),
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position].',
                                # 7x joint angles (q), (3 + 4)x FK (q), dq (7 joint angle velocities), 1x gripper state
                                # optional: delta fk (q) (avoid for now)
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                            # 3 deltas xyz 3 deltas roll pitch yaw 1 delta grip
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    # 'language_embedding': tfds.features.Tensor(
                    #     shape=(512,),
                    #     dtype=np.float32,
                    #     doc='Kona language embedding. '
                    #         'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    # ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""

        return {
            # change the path to match the datasets subfolder

            'train': self._generate_examples(path=f'../../../datasets/raw_data/{self.RELEASE_NAME}/episode_*.npz'),
            # 'val': self._generate_examples(path=f'../../../datasets/{self.RELEASE_NAME}val/episode_*.npz'),
        }

    def get_mp4_frames(self, mp4_path, resampled_indices):
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
        return np.array(frames)

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        print(f"Process ID: {os.getpid()}")

        def _parse_example(episode_path):

            # print(f"Processing episode: {episode_path}")

            # add deltas (from franka_pose --> split in xyz and rot_matrix)

            try:
                ep = EpisodeLogState.from_numpy(episode_path)
                ep.align_logs_with_resampling()
                # data.remove_near_zero_velocity_frames()
                data = ep.get_episode_data()
            except Exception as e:
                print(f"Error in processing episode {path}: {e}")
                return None

            primary_mp4_path = episode_path.replace('.npz', '.mp4').replace('episode_', 'primary_episode_')
            wrist_mp4_path = episode_path.replace('.npz', '.mp4').replace('episode_', 'wrist_episode_')

            primary_frames = self.get_mp4_frames(
                primary_mp4_path, resampled_indices=ep.primarycam_resampled_indices)
            wrist_frames = self.get_mp4_frames(
                wrist_mp4_path, resampled_indices=ep.wristcam_resampled_indices)

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end

            # TODO: check if what follows is correct
            episode = []
            for i in range(len(data)):

                # assuming the data is at 15Hz, resample to 5Hz
                # if i % 3 != 0:
                #     continue

                step = data[i]

                pose = step['franka_pose']
                pos = pose[:3, 3]
                grip = step['gripper_status']

                # compute Kona language embedding
                # language_embedding = self._embed([step['task_description']])[0].numpy()

                gripper_state = np.expand_dims(grip, axis=0)
                state = np.concatenate([step['franka_q'], pos, gripper_state]).astype(np.float32)

                # terminate_action = np.array([True if i == (len(data) - 1) else False], dtype=np.float32)
                action = step['action'].astype(np.float32)

                episode.append({
                    'observation': {
                        'primary_image': primary_frames[i],
                        'wrist_image': wrist_frames[i],
                        'state': state,
                    },
                    'action': action,
                    'discount': 1.0,
                    'reward': 1.0 if i == (len(data) - 1) else 0.0,
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': step['task_description']
                    # 'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # print(f"Processed episode: {episode_path}")
            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #     beam.Create(episode_paths)
        #     | beam.Map(_parse_example)
        # )
