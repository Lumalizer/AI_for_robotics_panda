import cv2
import numpy as np
import time
import threading

class BaseCamera:
    def __init__(self, is_recording: threading.Event = None):
        self.active = False
        self.frames = []
        self.logs = []
        self.time = []
        self.is_recording = is_recording

    def get_frame(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def stream_video_to_new_window(self):
        while True:
            frame = self.get_frame()
            if frame is not None:
                frame = np.array(frame)
                cv2.imshow('Video Feed', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        
    def clear_logs(self):
        self.logs = []
        self.time = []
        
    def camera_thread_fn(self):
        while True:
            while self.is_recording.is_set():
                try:
                    self.logs.append(self.get_frame())
                    self.time.append(time.time_ns())
                except Exception as e:
                    print(f"Error in camera thread: {e}")
                    self.stop()
                    raise                
            time.sleep(0.01)

    def start_camera_thread(self):
        try:
            self.cam_thread = threading.Thread(target=self.camera_thread_fn, daemon=True)
            self.cam_thread.start()
        except Exception as e:
            print(f"Camera not connected. {e}")
            raise