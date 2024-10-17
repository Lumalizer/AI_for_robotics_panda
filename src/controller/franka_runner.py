import numpy as np


class FrankaRunner:
    def __init__(self):
        pass

    def infer(self, obs, task) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError