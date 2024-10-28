"""
Must first run "server_ovla_octo.py" on a remote server, with port forwarding of port 8000.
"""



"""
import requests
import json_numpy
json_numpy.patch()
import numpy as np

import time
import zlib
import base64


img_with_history = np.ones((1,2, 256, 256, 3), dtype=np.uint8) # for octo; TODO: check OpenVLA, i think it's just (256,256,3) or (2,256,256,3)

for i in range(10):
    st = time.time()

    action = requests.post(
        "http://0.0.0.0:8000/act",
        json={"image": base64.b64encode(zlib.compress(img_with_history.tobytes())).decode('utf-8'), "instruction": "do something", "unnorm_key":"action"}  # TODO: unnorm key
    ).json()

    en = time.time()

    print('Total run time incl. network: ', en-st,  action[0])
"""
