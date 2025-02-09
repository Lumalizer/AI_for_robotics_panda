from controller.franka_controller import FrankaController
from options import Options

opts = Options(
    instruction="stack the blue block on the red block",
    subproject_folder="determine_octo_parameters",
    n_repetitions=20,
    
    execution_horizon=8,
    prediction_horizon=8,
    action_space="cartesian",

    mode="octo",
    model_type="small",
    max_seconds=60,
)


fc = FrankaController(opts)
# fc.collect_demonstrations()
fc.continually_run_from_server()
