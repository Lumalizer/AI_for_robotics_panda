# for configs.py
OXE_DATASET_CONFIGS = {
    "air_net": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        },


# for transforms.py
def air_net_transform_step(step: Dict[str, Any]) -> Dict[str, Any]:
    img = step['observation']['image']
    state = step['observation']['state']

    trajectory = {
        'observation': {
            'image': img,
            'state': state,
        },
        'action': step['action'],
    }

    for copy_key in ['discount', 'reward', 'is_first', 'is_last', 'is_terminal',
                     'language_instruction']:
        trajectory[copy_key] = step[copy_key]

    return trajectory

