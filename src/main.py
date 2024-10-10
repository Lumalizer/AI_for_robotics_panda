from controller.franka_controller import FrankaController
from controller.octo_runner import OctoRunner


fc = FrankaController(runner=OctoRunner())
fc.run_with_model()


# fc = FrankaController()
# fc.collect_demonstrations(10)