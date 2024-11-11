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
    def __init__(self, conversion_factor=0.003, angle_conversion_factor=0.4, 
                 mouse_axes_conversion=SpaceMouseState(1, 1, 1, 1, 1, 1), 
                 button_left_callback=do_nothing, button_right_callback=do_nothing):
        
        def button_callback(state, buttons):
            if buttons[0]:
                button_left_callback()
            if buttons[1]:
                button_right_callback()
        
        success = pyspacemouse.open(button_callback=button_callback)
        if not success:
            exit()

        self.conversion_factor = conversion_factor
        self.angle_conversion_factor = angle_conversion_factor
        self.mouse_axes_conversion = mouse_axes_conversion

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
            
        x = x * self.conversion_factor
        y = y * self.conversion_factor
        z = z * self.conversion_factor
        
        roll = roll * self.angle_conversion_factor
        pitch = pitch * self.angle_conversion_factor
        yaw = yaw * self.angle_conversion_factor
        
        state = SpaceMouseState(x, y, z, roll, pitch, yaw)

        return state*self.mouse_axes_conversion


    def close(self):
        pyspacemouse.close()

if __name__ == "__main__":
    smc = SpaceMouseController()
    while True:
        print(smc.read())
    smc.close()