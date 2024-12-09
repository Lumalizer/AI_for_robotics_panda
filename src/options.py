from dataclasses import dataclass
import os

@dataclass
class Options:
    instruction: str
    mode: str
    
    execution_horizon: int
    prediction_horizon: int = None
    
    subproject_folder: str = '' # "experiment" subfolder
    model_type: str = None
    
    xyz_multiplier: float = 0.03
    angle_multiplier: float = 15
    
    step_duration_s: float = 1/30
    action_multiplier: float = 1.0
    
    fps: int = 30
    unnorm_key: str = None
    randomize_starting_position: bool = True

    enable_pre_control: bool = True
    log: bool = True
    
    n_repetitions: int = 20
    max_seconds: int = 60
    ip: str = "http://0.0.0.0:8000/act"
    
    window_size: int = 2
    proprio: bool = False
    
    def __post_init__(self):
        assert(self.mode in ["octo", "openvla"])

        if self.mode == "octo":
            self.unnorm_key = "action"
        elif self.mode == "openvla":
            self.unnorm_key = "air_net"
            
    @property
    def save_folder(self):
        folder = os.path.join(self.subproject_folder, self.instruction)
        return folder