import os
import numpy as np
import FreeSimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from collections import Counter


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def clear_canvas(canvas):
    for child in canvas.winfo_children():
        child.destroy()

def update_plot_and_description(selected_dataset, selected_episode):
    data = np.load(f'datasets/{selected_dataset}/{selected_episode}', allow_pickle=True)
    keys = list(data[0].keys())
    
    task_description = data[0].get('task_description', 'No description available')
    window['task_description'].update(task_description)
    
    plottable_keys = [key for key in keys if key not in ["task_description", "franka_pose", "franka_t", "gripper_t", "camera_frame_t", "wrist_frame_t"]]
    
    fig, axes = plt.subplots(len(plottable_keys), 1, figsize=(10, len(plottable_keys) * 3))
    fig.tight_layout(pad=3.0)
    
    for ax, key in zip(axes, plottable_keys):
        values = [d[key] for d in data]
        if np.array(values).ndim > 2:
            for i in range(np.array(values).shape[1]):
                ax.plot([v[i] for v in values], label=f'{key}_{i}')
            ax.legend()
        else:
            ax.plot(values)
        ax.set_title(key)
    
    clear_canvas(window['-CANVAS-'].TKCanvas)
    draw_figure(window['-CANVAS-'].TKCanvas, fig)

def get_task_descriptions(selected_dataset):
    episodes = [f for f in os.listdir(f'datasets/{selected_dataset}') if f.endswith('.npy')]
    task_descriptions = []
    for episode in episodes:
        data = np.load(f'datasets/{selected_dataset}/{episode}', allow_pickle=True)
        task_description = data[0].get('task_description', 'No description available')
        task_descriptions.append(task_description)
    return Counter(task_descriptions)

available_datasets = sorted(os.listdir('datasets'))

# Define layout
layout = [
    [sg.Column([
        [sg.Text('Available Datasets')],
        [sg.Listbox(values=available_datasets, size=(20, 12), key='dataset_list', enable_events=True)],
        [sg.Text('Available Episodes')],
        [sg.Listbox(values=[], size=(20, 12), key='episode_list', enable_events=True)],
        [sg.Text('Task Description')],
        [sg.Multiline(size=(40, 5), key='task_description', disabled=True)],
        [sg.Text('Task Descriptions')],
        [sg.Listbox(values=[], size=(40, 12), key='task_description_list', enable_events=True)],
        [sg.Button('Clear Selection')]
    ]),
    sg.VSeperator(),
    sg.Column([
        [sg.Canvas(key='-CANVAS-')]
    ])],
    [sg.Button('Exit')]
]

window = sg.Window('Dataset Explorer', layout, finalize=True)

# Automatically select the first dataset and episode
if available_datasets:
    selected_dataset = available_datasets[0]
    episodes = [f for f in os.listdir(f'datasets/{selected_dataset}') if f.endswith('.npy')]
    episodes = sorted(episodes, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if episodes:
        selected_episode = episodes[0]
        update_plot_and_description(selected_dataset, selected_episode)
        window['dataset_list'].update(set_to_index=0)
        window['episode_list'].update(episodes, set_to_index=0)
        
        # Update task descriptions
        task_descriptions = get_task_descriptions(selected_dataset)
        task_description_list = sorted([f'{desc} ({count})' for desc, count in task_descriptions.items()])
        window['task_description_list'].update(task_description_list)

# Event loop
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    elif event == 'dataset_list':
        selected_dataset = values['dataset_list'][0]
        episodes = [f for f in os.listdir(f'datasets/{selected_dataset}') if f.endswith('.npy')]
        episodes = sorted(episodes, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        window['episode_list'].update(episodes)
        
        # Update task descriptions
        task_descriptions = get_task_descriptions(selected_dataset)
        task_description_list = sorted([f'{desc} ({count})' for desc, count in task_descriptions.items()])
        window['task_description_list'].update(task_description_list)
    elif event == 'episode_list':
        selected_episode = values['episode_list'][0]
        update_plot_and_description(selected_dataset, selected_episode)
    elif event == 'task_description_list':
        selected_task_description = values['task_description_list'][0].split(' (')[0]
        episodes = [f for f in os.listdir(f'datasets/{selected_dataset}') if f.endswith('.npy')]
        filtered_episodes = []
        for episode in episodes:
            data = np.load(f'datasets/{selected_dataset}/{episode}', allow_pickle=True)
            task_description = data[0].get('task_description', 'No description available')
            if task_description == selected_task_description:
                filtered_episodes.append(episode)
        window['episode_list'].update(sorted(filtered_episodes, key=lambda x: int(x.split('_')[-1].split('.')[0])))
    elif event == 'Clear Selection':
        episodes = [f for f in os.listdir(f'datasets/{selected_dataset}') if f.endswith('.npy')]
        episodes = sorted(episodes, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        window['episode_list'].update(episodes)

window.close()