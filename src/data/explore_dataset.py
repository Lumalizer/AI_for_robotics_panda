import os
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import FreeSimpleGUI as sg
import matplotlib.pyplot as plt
from collections import Counter
from episode_logstate import EpisodeLogState
import cv2
import threading
import time


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def clear_canvas(canvas):
    for child in canvas.winfo_children():
        child.destroy()


def update_plot_and_description(selected_dataset, selected_episode):
    data = EpisodeLogState.from_numpy(f'datasets/raw_data/{selected_dataset}/{selected_episode}')
    keys = list(data.__dict__.keys())

    task_description = data.task_description or 'No description available'
    window['task_description'].update(task_description)

    excluded_keys = ["task_description", "franka_pose", "franka_t", "gripper_t",
                     "camera_frame_t", "wrist_frame_t", "aligned", "filtered_nearzero_velocity"]

    plottable_keys = [key for key in keys if key not in excluded_keys]

    fig, axes = plt.subplots(len(plottable_keys), 1, figsize=(10, len(plottable_keys) * 3))
    fig.tight_layout(pad=3.0)

    for ax, key in zip(axes, plottable_keys):
        values = getattr(data, key)
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
    episodes = [f for f in os.listdir(f'datasets/raw_data/{selected_dataset}') if f.endswith('.npz')]
    task_descriptions = []
    for episode in episodes:
        data = EpisodeLogState.from_numpy(f'datasets/raw_data/{selected_dataset}/{episode}')
        task_description = data.task_description or 'No description available'
        task_descriptions.append(task_description)
    return Counter(task_descriptions)


video_threads = []


def stop_videos():
    global video_threads
    for thread in video_threads:
        if thread.is_alive():
            thread.do_run = False
    video_threads = []


def play_video(video_path, canvas_key):
    t = threading.currentThread()
    cap = cv2.VideoCapture(video_path)
    print(f'Playing video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps if fps > 0 else 1 / 30  # Default to 30 FPS if FPS is not available
    print(f'FPS: {fps}, Frame time: {frame_time}, n_frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')

    while cap.isOpened() and getattr(t, "do_run", True):
        ret, frame = cap.read()
        if not ret:
            break
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window.write_event_value(f'-UPDATE-{canvas_key}-', imgbytes)
        time.sleep(frame_time)  # Control frame rate
    cap.release()


def update_video(canvas_key, imgbytes):
    window[canvas_key].update(data=imgbytes)


def initialize_window(available_datasets):

    layout = [
        [
            sg.Column([
                [sg.Text('Selected Dataset')],
                [sg.Text('', size=(20, 1), key='selected_dataset')],
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
                [sg.Canvas(key='-CANVAS-', size=(640, 480))],
            ]),
            sg.Column([
                [sg.Image(key='-PRIMARY_VIDEO-', size=(640, 480))],
                [sg.Image(key='-WRIST_VIDEO-', size=(640, 480))]
            ]),
        ],
        [sg.Button('Exit')]
    ]
    window = sg.Window('Dataset Explorer', layout, resizable=True, finalize=True)
    return window


def auto_select_first_dataset_and_episode(available_datasets):
    if available_datasets:
        selected_dataset = available_datasets[0]
        window['selected_dataset'].update(selected_dataset)
        episodes = [f for f in os.listdir(f'datasets/raw_data/{selected_dataset}') if f.endswith('.npz')]
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

            # Play videos
            play_videos(selected_dataset, selected_episode)


def play_videos(selected_dataset, selected_episode):
    primary_thread = threading.Thread(target=play_video, args=(
        f'datasets/raw_data/{selected_dataset}/primary_{selected_episode.replace(".npz", ".mp4")}', '-PRIMARY_VIDEO-'), daemon=True)
    wrist_thread = threading.Thread(target=play_video, args=(
        f'datasets/raw_data/{selected_dataset}/wrist_{selected_episode.replace(".npz", ".mp4")}', '-WRIST_VIDEO-'), daemon=True)
    primary_thread.start()
    wrist_thread.start()
    video_threads.extend([primary_thread, wrist_thread])


def handle_events(event, values):
    selected_dataset = window['selected_dataset'].DisplayText
    if event in (sg.WIN_CLOSED, 'Exit'):
        return False
    elif event == 'dataset_list':
        selected_dataset = values['dataset_list'][0]
        window['selected_dataset'].update(selected_dataset)
        episodes = [f for f in os.listdir(f'datasets/raw_data/{selected_dataset}') if f.endswith('.npz')]
        episodes = sorted(episodes, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        window['episode_list'].update(episodes)

        # Update task descriptions
        task_descriptions = get_task_descriptions(selected_dataset)
        task_description_list = sorted([f'{desc} ({count})' for desc, count in task_descriptions.items()])
        window['task_description_list'].update(task_description_list)
    elif event == 'episode_list':
        stop_videos()
        selected_episode = values['episode_list'][0]
        update_plot_and_description(selected_dataset, selected_episode)

        # Play videos
        play_videos(selected_dataset, selected_episode)
    elif event == 'task_description_list':
        selected_task_description = values['task_description_list'][0].split(' (')[0]
        episodes = [f for f in os.listdir(f'datasets/raw_data/{selected_dataset}') if f.endswith('.npz')]
        filtered_episodes = []
        for episode in episodes:
            data = EpisodeLogState.from_numpy(f'datasets/raw_data/{selected_dataset}/{episode}')
            if data.task_description == selected_task_description:
                filtered_episodes.append(episode)
        window['episode_list'].update(sorted(filtered_episodes, key=lambda x: int(x.split('_')[-1].split('.')[0])))
    elif event == 'Clear Selection':
        episodes = [f for f in os.listdir(f'datasets/raw_data/{selected_dataset}') if f.endswith('.npz')]
        episodes = sorted(episodes, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        window['episode_list'].update(episodes)
    elif event.startswith('-UPDATE-'):
        canvas_key = event.split('-')[3]
        update_video(f'-{canvas_key}-', values[event])
    return True


available_datasets = sorted([dir for dir in os.listdir('datasets/raw_data')
                             if os.path.isdir(f'datasets/raw_data/{dir}')])
window = initialize_window(available_datasets)
auto_select_first_dataset_and_episode(available_datasets)

# Event loop
while True:
    event, values = window.read()
    if not handle_events(event, values):
        break

window.close()
