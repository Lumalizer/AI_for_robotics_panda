import cv2
from in_out.camera.BaseCamera import BaseCamera

# Set the FOV / zoom using v4l2-ctl
# sudo apt install v4l-utils
# find devices with "v4l2-ctl --list-devices"

class LogitechCamera(BaseCamera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active = False
        
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        cap = cv2.VideoCapture(9)
        if not cap.isOpened():
            raise Exception('Logitech camera not found')
        
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # we will not receive the exact aspect ratio that we want
        # so need to crop / resize later
        
        self.cap = cap
        
    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = self.crop_and_resize(frame, 256)
        return frame

    def start(self):
        if not self.active:
            self.active = True
            print(f'{self.name} camera started\n')
            self.get_frame()
            
    def stop(self):
        if self.active:
            self.active = False
            self.cap.release()
            print(f'{self.name} camera stopped\n')
            
        
if __name__ == '__main__':
    cam = LogitechCamera()
    cam.start()
    cam.stream_video_to_new_window()
    