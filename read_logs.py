import os
import pickle
import pandas as pd

path = "examples/sample_logs"
folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

for folder in folders:
    # print(folder)
    files = os.listdir(os.path.join(path, folder))
    logs = pd.read_csv(os.path.join(path, folder, "logs.csv"))
    trajectories = pickle.load(open(os.path.join(path, folder, "trajectory.pkl"), "rb"))

    print(logs.head())
    time, q, dq, poses = trajectories

    print(time[:5])

    # need to sync the gripper with the other logs (use time from both)
    print(len(time))
    print(len(logs))
