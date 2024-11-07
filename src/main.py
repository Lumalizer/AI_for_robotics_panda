from controller.franka_controller import FrankaController


fc = FrankaController(dataset_name="hover_blue_logi3")
# fc.collect_demonstrations(50)
fc.run_from_server()
