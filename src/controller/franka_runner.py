import numpy as np
import queue
import multiprocessing
import time

class FrankaRunner:
    def __init__(self):
        self.model_in_queue = multiprocessing.Queue()
        self.model_out_queue = multiprocessing.Queue()
        self.is_running = multiprocessing.Event()
        
        self.start_model_process()

    def infer(self, obs, task) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def model_process_fn(self):
        while True:
            if not self.model_in_queue.empty():
                self.is_running.set()
                obs, task = self.model_in_queue.get()
                print("Model process got input")
                
                action = self.infer(obs, task)
                print("Model process got output")
                self.model_out_queue.put(action)
                self.is_running.clear()
                
            time.sleep(0.01)
            
    def start_model_process(self):
        try:
            self.model_thread = multiprocessing.Process(target=self.model_process_fn, daemon=True)
            self.model_thread.start()
        except Exception as e:
            print(f"Model process not connected. {e}")
            raise