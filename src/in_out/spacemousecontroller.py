import pyspacemouse
from dataclasses import dataclass
import time
import threading

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
    
    @property
    def delta_pos(self):
        return [-self.y, self.x, self.z]
    
    @property
    def delta_rot(self):
        return [self.yaw, self.pitch, -self.roll]
    
def button_callback(state, buttons):
    print(f"State: {state}, Buttons: {buttons}")
    
def do_nothing():
    pass

class SpaceMouseController:
    def __init__(self, xyz_multiplier=0.003, angle_multiplier=0.4, 
                 button_left_callback=do_nothing, button_right_callback=do_nothing):
        
        def button_callback(state, buttons):
            if buttons[0]:
                button_left_callback()
            if buttons[1]:
                button_right_callback()
        
        try:
            pyspacemouse.open(button_callback=button_callback)
        except Exception as e:
            print("SpaceMouse not connected.")
            return None
            
        self.latest_state = pyspacemouse.read()
        self.update_latest_state_thread = threading.Thread(target=self.update_latest_state_thread_fn, daemon=True)
        self.update_latest_state_thread.start()

        self.conversion_factor = xyz_multiplier
        self.angle_conversion_factor = angle_multiplier
        
    def update_latest_state_thread_fn(self):
        while True:
            self.latest_state = pyspacemouse.read()
            time.sleep(0.01)

    def read(self) -> SpaceMouseState:
        state = self.latest_state

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
            
        x = x * self.conversion_factor
        y = y * self.conversion_factor
        z = z * self.conversion_factor
        
        roll = roll * self.angle_conversion_factor
        pitch = pitch * self.angle_conversion_factor
        yaw = yaw * self.angle_conversion_factor
        
        state = SpaceMouseState(x, y, z, roll, pitch, yaw)

        return state

    def close(self):
        pyspacemouse.close()

if __name__ == "__main__":
    smc = SpaceMouseController()
    while True:
        print(smc.read())
    smc.close()