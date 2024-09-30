import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = "logs/trajectory_2024-09-27_17-02-19"

data = pickle.load(open(os.path.join(path, "trajectory.pkl"), "rb"))

franka_t = np.squeeze(data["franka_t"])
franka_q = data["franka_q"]
franka_dq = data["franka_dq"]
franka_pose = data["franka_pose"]
gripper_t = data["gripper_t"]
gripper_status = data["gripper_status"]
camera_frame_t = data["camera_frame_t"]


# now we re-sample gripper (gripper_status / gripper_t) and franka_* to camera_frame_t (i.e., 30 Hz) by looking for the indices whose timestamp is closest to the camera_frame_t
franka_resampled_indices = np.searchsorted(franka_t, camera_frame_t)
gripper_resampled_indices = np.searchsorted(gripper_t, camera_frame_t)

if franka_resampled_indices[-1] == len(franka_t):
    franka_resampled_indices[-1] = len(franka_t) - 1
if gripper_resampled_indices[-1] == len(gripper_t):
    gripper_resampled_indices[-1] = len(gripper_t) - 1

franka_t = franka_t[franka_resampled_indices]
franka_q = franka_q[franka_resampled_indices]
franka_dq = franka_dq[franka_resampled_indices]
franka_pose = franka_pose[franka_resampled_indices]

gripper_t = gripper_t[gripper_resampled_indices]
gripper_status = gripper_status[gripper_resampled_indices]


# Now we have all the data we need, timestamp-aligned and sub-sampled at the same frame rate as the camera (ideally, 30Hz, if data collected through
# usb-3 port)

# e.g., a dataset for diffusion policy may have inputs/state as (img{t}, franka_q{t}, franka_dq{t} or img{t}, franka_pose{t}) and output as (franka_pose {t+1 : }, gripper_status {t+1 : })

x = franka_pose[:, 0, 3]
y = franka_pose[:, 1, 3]
z = franka_pose[:, 2, 3]

# plot traj in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.scatter(x,y,z)
# add limits
ax.set_xlim([0, 0.9])
ax.set_ylim([-0.5, 0.5])
ax.set_zlim([0, 0.8])

plt.show()


