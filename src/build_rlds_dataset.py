import os
import pickle

episodes = os.listdir('logs')

for episode_path in episodes:
    
    trajectory_path = os.path.join('logs', episode_path, 'trajectory.pkl')
    video_path = os.path.join('logs', episode_path, 'video.mp4')
    
    with open(trajectory_path, 'rb') as f:
        trajectory = pickle.load(f)
        
    print(f"Trajectory: {episode_path}, Length: {len(trajectory)} steps, Keys: {trajectory[0].keys()}")