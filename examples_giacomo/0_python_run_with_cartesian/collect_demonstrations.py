
import numpy as np
import pickle
from pynput import keyboard
import time

from panda_py import constants
constants.JOINT_LIMITS_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
constants.JOINT_LIMITS_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
import panda_py
from panda_py import libfranka

panda = panda_py.Panda("172.16.0.2")
gripper = libfranka.Gripper("172.16.0.2")
gripper.stop()

panda.move_to_start()


MAX_LEN = 60
print('Press <space> to start/finish recording a trajectory. Press <s> to save all trajectories and exit. Trajectories have a maximum length of ', MAX_LEN, 'seconds, after which they automatically terminate.')

trajectories = []

panda.teaching_mode(True)
is_recording = False
recording_start_t = 0

def process_log():
    q = panda.get_log()['q']
    dq = panda.get_log()['dq']


    poses = []
    for qq in q:
        poses.append(panda_py.fk(qq))
    return [np.array(q), np.array(dq), np.array(poses)]


def on_release(key):
    global is_recording, recording_start_t
    if key == keyboard.Key.space:
        if is_recording:
           is_recording = False
           print('Trajectory recorded!')
           panda.disable_logging()
           trajectories.append( process_log() )
        else:
            is_recording = True
            print('Recording trajectory...')
            panda.enable_logging(MAX_LEN * 1000)
            recording_start_t = time.time()
    elif 'char' in dir(key) and key.char=='s':
        print('Saving trajectories to file!')
        with open('out_trajectories.pkl', 'wb') as f:
            pickle.dump(trajectories, f)

listener = keyboard.Listener(on_release=on_release)
listener.start()

while True:
    #if is_recording:
    #

    ## TODO: still missing, logging the gripper state!!   i can read it as gripper.read_once(), but it's not automatic as is logging
    # TODO: make sure the gripper readings are aligned with the other logs, or else just take out the other logs yourself

    if time.time()-recording_start_t > MAX_LEN and is_recording:
        print('Timeout. Trajectory recorded!')
        panda.disable_logging()  
        trajectories.append( process_log() )
    time.sleep(0.01)
