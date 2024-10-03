import numpy as np

path = '/home/air-realtime/Documents/GitHub/AI_for_robotics_panda/logs/trajectory_2024-10-03_11-14-19/trajectory.npy'
data = np.load(path, allow_pickle=True)

print(data[0].keys())

# check the dimensions of the data
for i, step in enumerate(data):
    # print(f"Step {i}: {step.keys()}")
    # print(f"Step {i}: {step['image'].shape}")
    # print(f"Step {i}: {step['franka_q'].shape}")
    # print(f"Step {i}: {step['gripper_status'].shape}")
    # grips = np.expand_dims(step['gripper_status'], axis=0)
    # grips = np.concatenate([grips, grips, [0]])
    # print(f"Step {i}: {grips}")
    # state = np.concatenate([step['franka_q'], grips, grips, grips]).astype(np.float32)
    # print(f"Step {i}: {state.shape}")
    # print(f"Step {i}: {step['franka_dq'].shape}")
    # print(f"Step {i}: {step['franka_pose']}")
    print()