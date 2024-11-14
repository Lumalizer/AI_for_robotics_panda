from controller.franka_controller import FrankaController


fc = FrankaController(dataset_name="octo_with_wrist_fixed")
# fc.collect_demonstrations(50)
fc.continually_run_from_server()