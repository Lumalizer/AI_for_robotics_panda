from controller.franka_controller import FrankaController


fc = FrankaController(dataset_name="grasp_blue_300", receding_horizon=None, randomize_starting_position=True)
fc.collect_demonstrations(50)
# fc.continually_run_from_server()