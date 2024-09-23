import pyspacemouse
import time

success = pyspacemouse.open()
if not success:
    exit()

rate = 100

while True:
    assert 0 < rate <= 100 # rate must be between 1 and 100

    start = time.time()
    state = pyspacemouse.read()

    # axis -> positive / negative
    # y -> forward / backward
    # x -> right / left
    # z -> pull up / push down

    # roll -> rotate right / left (y-axis)
    # pitch -> rotate forward / backward (x-axis)
    # yaw -> rotate right / left (z-axis)

    x, y, z = state.x, state.y, state.z
    roll, pitch, yaw = state.roll, state.pitch, state.yaw

    while time.time()-start < 1/rate:
        time.sleep(0.01)

    print(f"Took {round((time.time()-start), 3)} seconds. {state}")
    