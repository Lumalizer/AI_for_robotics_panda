import cv2
from in_out.camera.BaseCamera import BaseCamera

# Set the FOV / zoom using v4l2-ctl
# sudo apt install v4l-utils

class LogitechCamera(BaseCamera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active = False
        
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # we will not receive the exact aspect ratio that we want
        # so need to crop / resize later
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.aspect_diff = actual_width - actual_height
        
        self.cap = cap
        
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = frame[:, int(self.aspect_diff/2):int(-self.aspect_diff/2)]
        frame = cv2.resize(frame, (256, 256))
        return frame

    def start(self):
        if not self.active:
            self.active = True
            print('Camera started')
            self.get_frame()
            
    def stop(self):
        if self.active:
            self.active = False
            self.cap.release()
            print('Camera stopped')
            
        
if __name__ == '__main__':
    cam = LogitechCamera()
    cam.start()
    cam.stream_video_to_new_window()
    