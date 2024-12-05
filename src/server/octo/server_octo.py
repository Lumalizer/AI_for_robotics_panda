"""
server_octo.py

Provide a lightweight server/client implementation for deploying octo models over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs octo model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

import zlib
import base64

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": base64.b64encode(zlib.compress(img_with_history.tobytes())).decode('utf-8'), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 USER@<SERVER_IP>`
"""

# ruff: noqa: E402
import jax
from fastapi.responses import JSONResponse
from fastapi import FastAPI
import uvicorn
import torch
import draccus
from typing import Any, Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import traceback
import logging
import json
from octo.model.octo_model import OctoModel
import numpy as np
import base64
import zlib
import time
import json_numpy
json_numpy.patch()
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--octo_path', type=str, required=True)
    parser.add_argument('--window_size', type=int, default=1)
    return parser.parse_args()

class OctoServer:
    def __init__(self, octo_path: Union[str, Path], window_size: int = 1):
        """
        A simple server for Octo models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.octo_path = octo_path
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.model = OctoModel.load_pretrained(self.octo_path)  # TODO: device=self.device)
        print(self.model.get_pretty_spec())

        self.previous = {}
        
        if not window_size == 1 or window_size == 2:
            raise ValueError("Only window sizes of 1 or 2 are supported.")
        
        self.window_size = window_size

    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            st = time.time()

            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            primary_image, wrist_image, instruction = payload["primary_image"], payload["wrist_image"], payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)

            primary_image = np.frombuffer(zlib.decompress(base64.b64decode(primary_image)), dtype=np.uint8).reshape((1, 1, 256, 256, 3))
            wrist_image = np.frombuffer(zlib.decompress(base64.b64decode(wrist_image)), dtype=np.uint8).reshape((1, 1, 128, 128, 3))

            if self.window_size == 2:
                if "primary_image" not in self.previous or "wrist_image" not in self.previous:
                    self.previous["primary_image"] = primary_image
                    self.previous["wrist_image"] = wrist_image

                window_primary_image = np.concatenate([self.previous["primary_image"], primary_image], axis=1)
                window_wrist_image = np.concatenate([self.previous["wrist_image"], wrist_image], axis=1)

                self.previous["primary_image"] = primary_image
                self.previous["wrist_image"] = wrist_image
            else:
                window_primary_image = primary_image
                window_wrist_image = wrist_image
            
            if isinstance(instruction, str):
                instruction = [instruction]

            # Run Octo Inference
            observation = {
                # 'primary_image': primary_image,
                # 'wrist_image': wrist_image,
                'image_primary': window_primary_image,
                'image_wrist': window_wrist_image,
                'timestep_pad_mask': np.full((1, window_primary_image.shape[1]), True, dtype=bool),
                # 'proprio': payload["state"]
            }
            task = self.model.create_tasks(texts=instruction)                  # for language conditioned

            actions = self.model.sample_actions(
                observation,
                task,
                unnormalization_statistics=self.model.dataset_statistics[unnorm_key] if unnorm_key else None,
                rng=jax.random.PRNGKey(0)
            )
            actions = np.array(actions[0])  # remove batch dim

            en = time.time()
            print('Inference time:', en-st)

            if double_encode:
                return JSONResponse(json_numpy.dumps(actions))
            else:
                return JSONResponse(actions)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)


# @dataclass
# class DeployConfig:
#     # fmt: off
#     model = "octo"
#     # octo_path = "/home/u950323/trained-models/octo_checkpoints/air_net"
#     octo_path = args.octo_path

#     # Server Configuration
#     host: str = "0.0.0.0"                                               # Host IP Address
#     port: int = 8000                                                    # Host Port
#     # fmt: on


@draccus.wrap()
def deploy(octo_path: str, window_size: int = 1) -> None:
    server = OctoServer(octo_path, window_size)
    server.run("0.0.0.0", port=8000)


if __name__ == "__main__":
    args = parse_args()
    deploy(args.octo_path, args.window_size)