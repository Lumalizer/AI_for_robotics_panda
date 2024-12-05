import sys
import os
from episode_logstate import EpisodeLogState

def update_task_description(dataset_path, description_to_replace, new_description):
    episodes = [f for f in os.listdir(dataset_path) if f.endswith('.npz')]

    for episode in episodes:
        data = EpisodeLogState.from_numpy(f'{dataset_path}/{episode}')
        
        if data.task_description == description_to_replace:
            print(f'Updating task description in {episode} from "{description_to_replace}" to "{new_description}"')
            data.task_description = new_description

            data.to_numpy(f'{dataset_path}/{episode}')

if __name__ == '__main__':
    # datasets = [
    #     f'datasets/raw_data/{f}' for f in sorted(os.listdir('datasets/raw_data')) if os.path.isdir(f'datasets/raw_data/{f}')]
    
    # for dataset in datasets:
    #     update_task_description(dataset, 'place the blue cube in the box', 'place the blue block in the box')


    # fixes to the task descriptions
    update_task_description('datasets/raw_data/100_stack_red_blue_100',
                            '', 'stack the red block on the blue block')
    
    update_task_description('datasets/raw_data/pack_box_50',
                            "place all of the items in the box",
                            "place all of the items into the box")
    
    update_task_description('datasets/raw_data/pick_up_blue_200',
                            "spick up the blue block",
                            "pick up the blue block")
    
    update_task_description('datasets/raw_data/unpack_box_50',
                            "50,remove all of the items from the box",
                            "remove all of the items from the box")

