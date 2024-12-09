from controller.franka_controller import FrankaController
from options import Options

opts = Options(
    instruction="iron the shirt",
    subproject_folder="ironing_test",
    n_repetitions=20,
    
    mode="octo",
    model_type="small",
    execution_horizon=8,
    prediction_horizon=16,
    max_seconds=60,
)


fc = FrankaController(opts)
# fc.collect_demonstrations()
fc.continually_run_from_server()
