from controller.franka_controller import FrankaController


fc = FrankaController(dataset_name="unpack_box_50", mode="octo", 
                      receding_horizon=16, randomize_starting_position=True)
# fc.collect_demonstrations(1)
fc.continually_run_from_server(instruction = "pick up the blue block")
