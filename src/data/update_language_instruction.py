import sys
import os
from data.episode_logstate import EpisodeLogState

def update_task_description(dataset_path, description_to_replace, new_description):
    episodes = [f for f in os.listdir(dataset_path) if f.endswith('.npz')]

    for episode in episodes:
        data = EpisodeLogState.from_numpy(f'{dataset_path}/{episode}')
        
        if data.task_description == description_to_replace:
            data.task_description = new_description

            data.to_numpy(f'{dataset_path}/{episode}')

if __name__ == '__main__':
    datasets = [
        f'datasets/raw_data/{f}' for f in sorted(os.listdir('datasets/raw_data')) if os.path.isdir(f'datasets/raw_data/{f}')]
    
    for dataset in datasets:
        update_task_description(dataset, 'place the blue cube in the box', 'place the blue block in the box')