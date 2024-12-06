from controller.franka_controller import FrankaController


fc = FrankaController(dataset_name="test_inference", mode="octo",
                      execution_horizon=8, randomize_starting_position=True)
# fc.collect_demonstrations(50)
fc.continually_run_from_server(instruction="pick up the iron", save=True)
