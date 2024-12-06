from controller.franka_controller import FrankaController


fc = FrankaController(dataset_name="test_inference", mode="octo",
                      execution_horizon=8, randomize_starting_position=True)

# fc.collect_demonstrations(50)

fc.continually_run_from_server(
    instruction="first knock over the bottle and then pick up the bottle", save=True,
    max_seconds=30)
