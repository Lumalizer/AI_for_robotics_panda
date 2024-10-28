from controller.franka_controller import FrankaController


fc = FrankaController()
# fc.collect_demonstrations(50)
fc.run_from_server()