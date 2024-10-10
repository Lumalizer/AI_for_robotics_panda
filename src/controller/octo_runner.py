from functools import partial
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)
from controller.franka_runner import FrankaRunner
from octo.model.octo_model import OctoModel
from octo.utils.train_callbacks import supply_rng
import jax


class OctoRunner(FrankaRunner):
    def __init__(self, model: str="trained_models"):
        self.model = OctoModel.load_pretrained(model, 4999)
        
        print(self.model.dataset_statistics.keys())
        
    def sample_actions(self, model, observations, tasks, rng, argmax=True, temperature=1.0, *args, **kwargs):
        # add batch dim to observations
        observations = jax.tree_map(lambda x: x[None], observations)
        actions = model.sample_actions(
            observations,
            tasks,
            rng=rng,
            unnormalization_statistics=model.dataset_statistics["action"],
        )
        # remove batch dim
        return actions[0]
    
    @property
    def policy_fn(self):
        return supply_rng(
            partial(
                self.sample_actions,
                self.model,
                argmax=True,
                temperature=1.0,
            )
        )

if __name__ == "__main__":
    from controller.franka_controller import FrankaController
    
    fc = FrankaController(runner=OctoRunner())
    fc.run_with_model()