from typing import Iterator, Tuple, Any
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import cv2
from scipy.spatial.transform import Rotation

class AirNet(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        # 'wrist_image': tfds.features.Image(
                        #     shape=(64, 64, 3),
                        #     dtype=np.uint8,
                        #     encoding_format='png',
                        #     doc='Wrist camera RGB observation.',
                        # ),
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
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
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
            
            'train': self._generate_examples(path='../../datasets/test_franka_ds/episode_*.npy'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        
        def _parse_example(episode_path):

            # TODO: add deltas (from franka_pose --> split in xyz and rot_matrix)

            data = np.load(episode_path, allow_pickle=True)  # list of dicts in our case
            mp4_path = episode_path.replace('.npy', '.mp4')
            
            # load mp4 and unpack frames into a np array using cv2 tin order to save up on space
            
            cap = cv2.VideoCapture(mp4_path)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                
            cap.release()
            
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            
            episode = []
            for i in range(len(data) - 1):
                step = data[i]
                next_step = data[i + 1]

                pose = step['franka_pose']
                pos = pose[:3, 3]
                rot = Rotation.from_matrix(pose[:3, :3])

                next_pose = next_step['franka_pose']
                next_pos = next_pose[:3, 3]
                next_rot = Rotation.from_matrix(next_pose[:3, :3])

                delta_xyz = next_pos - pos

                delta_rot = next_rot * rot.inv()
                delta_y, delta_p, delta_r = delta_rot.as_euler('zyx', degrees=False)

                grip = next_step['gripper_status']
                
                # compute Kona language embedding
                language_embedding = self._embed([step['task_description']])[0].numpy()

                gripper_state = np.expand_dims(grip, axis=0)
                state = np.concatenate([step['franka_q'], pos, gripper_state]).astype(np.float32)
                
                # terminate_action = np.array([True if i == (len(data) - 1) else False], dtype=np.float32)
                action = np.array([*delta_xyz, delta_y, delta_p, delta_r, grip]).astype(np.float32)
                
                episode.append({
                    'observation': {
                        'image': frames[i],
                        # 'wrist_image': step['wrist_image'],
                        'state': state, 
                    },
                    'action': action,
                    'discount': 1.0,
                    'reward': 1.0 if i == (len(data) - 1) else 0.0,
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': step['task_description'],
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

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
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

