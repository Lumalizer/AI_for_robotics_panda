# First import the library
import pyrealsense2 as rs
import numpy as np
from in_out.camera.BaseCamera import BaseCamera
import cv2

class RealSenseCamera(BaseCamera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

        self.config.enable_stream(rs.stream.color, 256, 256, rs.format.rgb8, 30)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        rgb = frames.get_color_frame()

        if not rgb: 
            return None

        rgb_data = rgb.as_frame().get_data()
        
        # need to copy to avoid filling the buffer and getting duplicate frames
        np_image = np.asanyarray(rgb_data)
        
        # convert to the right color format
        np_image = cv2.cvtColor(np_image.copy(), cv2.COLOR_BGR2RGB)

        return np_image
    
    def start(self):
        if not self.active:
            self.active = True
            self.pipeline.start()

            for i in range(5):
                frames = self.pipeline.wait_for_frames()
            print('Camera started')
    
    def stop(self):
        if self.active:
            self.active = False
            self.pipeline.stop()



if __name__ == "__main__":
    cam = RealSenseCamera()
    cam.start()
    cam.stream_video_to_new_window()