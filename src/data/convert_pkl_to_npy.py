import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from data.episode_logstate import EpisodeLogState

def transform_dataset_from_pickle_to_numpy(dataset_path):
    # list all files that end with .pkl and unpickle them
    episodes = [f for f in os.listdir(dataset_path) if f.endswith('.pkl')]

    for episode in episodes:
        data = EpisodeLogState.from_pickle(f'{dataset_path}/{episode}')
        data.to_numpy(f'{dataset_path}/{episode.replace(".pkl", "")}')
        os.remove(f'{dataset_path}/{episode}')

if __name__ == '__main__':
    datasets = [
        f'datasets/raw_data/{f}' for f in sorted(os.listdir('datasets/raw_data')) if os.path.isdir(f'datasets/raw_data/{f}')]
    
    for dataset in datasets:
        transform_dataset_from_pickle_to_numpy(dataset)