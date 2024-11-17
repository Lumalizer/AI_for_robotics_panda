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
        
        if len(devices) == 0:
            raise Exception('No RealSense devices found')
        
        for dev in devices:
            dev.hardware_reset()

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)

        self.config.enable_stream(stream_type=rs.stream.color, width=1280, height=720, format=rs.format.rgb8, framerate=self.fps)
        

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        rgb = frames.get_color_frame()

        if not rgb: 
            return None

        rgb_data = rgb.as_frame().get_data()
        
        # need to copy to avoid filling the buffer and getting duplicate frames
        np_image = np.asanyarray(rgb_data)

        # resize
        np_image = self.crop_and_resize(np_image, 256)
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

        return np_image
    
    def start(self):
        if not self.active:
            self.active = True
            self.pipeline.start()

            for i in range(5):
                frames = self.pipeline.wait_for_frames()
            print(f'{self.name} Camera started\n')
    
    def stop(self):
        if self.active:
            self.active = False
            self.pipeline.stop()
            print(f'{self.name} Camera stopped\n')



if __name__ == "__main__":
    cam = RealSenseCamera()
    cam.start()