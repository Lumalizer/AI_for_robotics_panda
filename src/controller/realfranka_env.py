import gymnasium as gym
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from panda_py import constants
constants.JOINT_LIMITS_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
constants.JOINT_LIMITS_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
import panda_py
from panda_py import controllers
from panda_py import libfranka
from multiprocessing import Process, Pipe

def gripper_controller_process(franka_ip, conn):
    gripper = libfranka.Gripper(franka_ip)
    while True:
        if conn.poll(0):
            cmd = conn.recv()
            if cmd[0] == 'open':
                gripper.move(0.08, 0.5)
            elif cmd[0] == 'close':
                # TODO : gripper.grasp should probably be used to pick up items of various size, but it is buggy (sometimes, gripper connection is lost completely)
                # gripper.grasp(0.02, 0.5, 100, .3, .3) # for ironing
                gripper.move(0.035, 0.5) # normal
            elif cmd[0] == 'read':
                grip = gripper.read_once().is_grasped
                conn.send(grip)
        time.sleep(0.01)
    

def franka_controller_process(franka_ip, action_space, conn, parent_conn_gripper, multiplier: int):
    fr3 = panda_py.Panda(franka_ip)

    if action_space == "cartesian":
        controller = controllers.CartesianImpedance()
    elif action_space == "joint":
        controller = controllers.JointPosition()
    else:
        print("CONTROLLER NOT IMPLEMENTED YET: ", action_space)
        return

    # TODO: implement joint controller
    
    def start_controller(controller=controller):
        # print(f"Starting {controller.__class__.__name__} controller")
        fr3.start_controller(controller)
        
    def stop_controller():
        # print("Stopping controller")
        fr3.stop_controller()
        
    start_controller()

    target_xyz = None
    target_quat = None
    target_q = None
    
    parent_conn_gripper.send(('read',))
    gripper_state = parent_conn_gripper.recv()
    gripper_action = gripper_state
    
    with fr3.create_context(frequency=1e3, max_runtime=-1) as ctx:
        while ctx.ok():
            if conn.poll(0): # non-blocking check if there is any command availble
                cmd = conn.recv() # tuple:  ('action', [....]) or ('get_state')
                if cmd[0] == 'action':
                    new_action = cmd[1]
                
                    if action_space == "cartesian":
                        # TODO: remove this hack
                        delta_x = new_action[:3] *multiplier # delta_x, delta_y, delta_z
                        delta_rot = new_action[3:6] *multiplier # delta_yaw, delta_pitch, delta_roll
                        gripper_action = new_action[6]
                        
                        if gripper_action >= 1.0:
                            gripper_action = 1
                        elif gripper_action <= 0.0:
                            gripper_action = 0
                        else:
                            gripper_action = gripper_state

                        pose = fr3.get_pose().copy()
                        x = pose[0:3, 3]
                        rot = R.from_matrix(pose[0:3, 0:3])
                        delta_rot = R.from_euler('zyx', delta_rot, degrees=True)

                        target_xyz = np.array(x + delta_x)
                        target_quat = (rot * delta_rot).as_quat()
                    elif action_space == "joint":
                        target_q = new_action.copy()[:7]
                        gripper_action = new_action[7]

                elif cmd[0] == 'move_to_start':
                    target_xyz = [3.08059678e-01, -1.82101039e-04,  4.86319269e-01]
                    target_quat = [0.99999257, -0.00275055,  0.00102066,  0.00249987]
                    target_q = [3.76427204e-02, -7.64296584e-01, -1.27612257e-02, -2.35961049e+00, -1.54984590e-02, 1.59347292e+00, 8.35692266e-01]

                    stop_controller()
                    fr3.move_to_start()
                    start_controller()
                    
                elif cmd[0] == 'get_state':
                    # 7x joint angles (q), (3+4)x FK (xyz, quaternion), 7x joint velocities (dq), 1x gripper state
                    state = fr3.get_state()

                    _pose = fr3.get_pose()
                    pos = _pose[:3, 3]
                    quat = R.from_matrix(_pose[:3, :3]).as_quat() # XYZ W

                    state = np.array([*state.q, *pos, *quat, *state.dq, gripper_state])
                    conn.send(state)
                    
                elif cmd[0] == 'stop_controller':
                    stop_controller()
                    
                elif cmd[0] == 'open_gripper':
                    gripper_state = gripper_action = 0
                    parent_conn_gripper.send(('open',))
                    
                elif cmd[0] == 'close_gripper':
                    gripper_state = gripper_action = 1
                    parent_conn_gripper.send(('close',))
                    
                elif cmd[0] == 'enable_logging':
                    fr3.enable_logging(cmd[1])
                    
                elif cmd[0] == 'disable_logging':
                    fr3.disable_logging()
                    
                elif cmd[0] == 'get_log':
                    log = fr3.get_log()
                    conn.send(log)

            if target_xyz is not None and target_quat is not None:
                # TODO: clip target_xyz to workspace volume

                if action_space == "cartesian":
                    controller.set_control(target_xyz, target_quat)
                elif action_space == "joint":
                    fr3.move_to_joint_position(target_q, speed_factor=0.3)
                    #controller.set_control(target_q, np.array([0.0001]*7))

            # execute gripper action
            if int(gripper_action) != int(gripper_state):
                if gripper_action == 1:
                    gripper_state = 1
                    parent_conn_gripper.send(('close',))
                elif gripper_action == 0:
                    gripper_state = 0
                    parent_conn_gripper.send(('open',))
            
    conn.close()


class RealFrankaEnv(gym.Env):
    def __init__(self, franka_ip="172.16.0.2", action_space="cartesian", multiplier: int=1, step_duration_s=-1):
        """
            step_duration_s = -1 means no rate limit is enforced; equivalent to blocking=False
            action_space = "cartesian" or "joint"
        """

        assert action_space in ["cartesian", "joint"]

        self.step_duration_s = step_duration_s

        # 7x joint angles (q), (3+4)x FK (xyz, quaternion), 7x joint velocities (dq), 1x gripper state
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7+3+4+7+1,), dtype=np.float32)

        self.parent_conn_gripper, self.child_conn_gripper = Pipe()
        self.gripper_process = Process(target=gripper_controller_process, args=(franka_ip, self.child_conn_gripper))
        self.gripper_process.start()
        
        self.parent_conn, self.child_conn = Pipe()
        self.controller_process = Process(target=franka_controller_process, args=(franka_ip, action_space, self.child_conn, self.parent_conn_gripper, multiplier))
        self.controller_process.start()
        
        if action_space == "cartesian":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        elif action_space == "joint":
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32) # joint + gripper
            
    def get_state(self):
        self.parent_conn.send(('get_state',))
        state = self.parent_conn.recv()
        return state
    
    def enable_logging(self, buffer_size: int):
        self.parent_conn.send(('enable_logging', buffer_size))
        
    def disable_logging(self):
        self.parent_conn.send(('disable_logging',))
        
    def get_log(self):
        self.parent_conn.send(('get_log',))
        log = self.parent_conn.recv()
        return log
    
    def stop_controller(self):
        self.parent_conn.send(('stop_controller',))
    
    def open_gripper(self):
        self.parent_conn.send(('open_gripper',))
        
    def close_gripper(self):
        self.parent_conn.send(('close_gripper',))

    def reset(self, seed=None, **kwargs):
        self.open_gripper()
        self.parent_conn.send(('move_to_start',))
        time.sleep(2)

        state = self.get_state()
        return state, {}

    def step(self, action):
        self.parent_conn.send(('action', action))

        if self.step_duration_s > 0:
            time.sleep(self.step_duration_s)

        state = self.get_state()
        return state, 0, False, False, {}

    def close(self):
        self.controller_process.terminate()
        self.controller_process.join()




if __name__ == "__main__":
    env = RealFrankaEnv(step_duration_s=0.2, action_space="cartesian")
    state, _ = env.reset()

    xyz = np.array([0,0,-0.1, 0,0,0, 0])
    print(state)

    for i in range(5):
        #action = env.action_space.sample()
        if i==0:
            action = xyz
        else:
            action = xyz*0

        #action = np.array([3.57426000e-02, -7.31155348e-01, -1.32127258e-02, -2.53634033e+00, -1.41506781e-02,  1.71452198e+00,  8.25908782e-01, 0])
        state, reward, done, truncated, info = env.step(action)

    env.close()


#useful wrappers:
#from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
#                    # from openvla:  for 15hz control they predict T=16 steps and execute X=8 steps (533ms);   for 5hz they predict T=9 and exec X=3 (600ms)
