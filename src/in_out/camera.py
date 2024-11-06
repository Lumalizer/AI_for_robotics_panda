# First import the library
import pyrealsense2 as rs
import numpy as np
import time
import cv2

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

        self.active = False
        # self.pipeline.start()

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
            
    def stream_video_to_new_window(self):
        while True:
            frame = self.get_frame()
            if frame is not None:
                frame = np.array(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow('Video Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cv2.destroyAllWindows()




"""
To check actual frame rate;  we set 30fps but this only works if your laptop has a usb3 (blue) port, if not it sets it to 15hz which is not good enough for us.
"""
if __name__ == "__main__":
    cam = Camera()
    cam.start()
    cam.stream_video_to_new_window()