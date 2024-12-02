from controller.franka_controller import FrankaController


fc = FrankaController(dataset_name="test", mode="demonstration", 
                      receding_horizon=None, randomize_starting_position=True)
fc.collect_demonstrations(100)
# fc.continually_run_from_server(instruction = "pick up the blue block")
