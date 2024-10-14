# First import the library
import pyrealsense2 as rs
import numpy as np
from PIL import Image

class Camera:
    def __init__(self):
        # reset devices to fix errors
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)

        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        self.pipeline.start()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        rgb = frames.get_color_frame()
        
        # print("camera frame (should increase): ", rgb.get_frame_number())

        if not rgb: 
            return None

        rgb_data = rgb.as_frame().get_data()
        
        # need to copy to avoid filling the buffer and getting duplicate frames
        np_image = np.asanyarray(rgb_data)

        return np_image.copy()
    
    def stop(self):
        self.pipeline.stop()




"""
To check actual frame rate;  we set 30fps but this only works if your laptop has a usb3 (blue) port, if not it sets it to 15hz which is not good enough for us.
"""
if __name__ == "__main__":
    import time
    cam = Camera()
    for i in range(100):
        st = time.time()
        img = cam.get_frame()
        en = time.time()
        print(f"Time taken to get frame: {en-st} s,  ", 1./(en-st), "fps")
        