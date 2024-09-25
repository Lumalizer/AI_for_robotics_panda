import pyspacemouse
from dataclasses import dataclass
import time

# axis -> positive / negative
# y -> forward / backward
# x -> right / left
# z -> pull up / push down

# roll -> rotate right / left (y-axis)
# pitch -> rotate forward / backward (x-axis)
# yaw -> rotate right / left (z-axis)

@dataclass
class SpaceMouseState:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    def __mul__(self, other):
        return SpaceMouseState(self.x*other.x, self.y*other.y, self.z*other.z, self.roll*other.roll, self.pitch*other.pitch, self.yaw*other.yaw)

class SpaceMouseController:
    def __init__(self):
        success = pyspacemouse.open()
        if not success:
            exit()

        self.rate_limit = 100
        self.angle_conversion_factor = 1
        self.conversion_factor = 0.01
        self.last_read = time.time()

    def read(self) -> SpaceMouseState:
        state = pyspacemouse.read()

        x, y, z = state.x, state.y, state.z
        roll, pitch, yaw = state.roll, state.pitch, state.yaw

        if abs(x)<0.05:
            x = 0
        if abs(y)<0.05:
            y = 0
        if abs(z)<0.05:
            z = 0

        if abs(roll)<0.05:
            roll = 0
        if abs(pitch)<0.05:
            pitch = 0
        if abs(yaw)<0.05:
            yaw = 0

        while time.time()-self.last_read < 1/self.rate_limit:
            time.sleep(0.005)

        self.last_read = time.time()

        return SpaceMouseState(x, y, z, roll, pitch, yaw)


    def close(self):
        pyspacemouse.close()

if __name__ == "__main__":
    smc = SpaceMouseController()
    while True:
        print(smc.read())
    smc.close()