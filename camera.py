# First import the library
import pyrealsense2 as rs
import numpy as np
from PIL import Image

class Camera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.pipeline.start()

    def get_frame(self):
           
        frames = self.pipeline.wait_for_frames()
        rgb = frames.get_color_frame()

        if not rgb: 
            return None

        rgb_data = rgb.as_frame().get_data()
        np_image = np.asanyarray(rgb_data)

        img = Image.fromarray(np_image)
        return img

    def stop(self):
        self.pipeline.stop()
