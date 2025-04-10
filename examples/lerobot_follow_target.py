import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import math
from pynput import keyboard
from lerobot_kinematics import lerobot_IK_5DOF, lerobot_FK_5DOF, get_robot

np.set_printoptions(linewidth=200)
os.environ["MUJOCO_GL"] = "egl"

# Load Model
xml_path = "./examples/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
mjdata = mujoco.MjData(mjmodel)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]])
        # Set joint limits

JOINT_INCREMENT, POSITION_INCREMENT = 0.005, 0.0008
control_qlimit = [[-2.8, -3.8, -0.4, -2, -3.14, -0.15], [2.8, 0.3, 3.4, 2, 3.14, 1.5]]
control_glimit = [[0.025, -0.4, 0.046, -3.1, -0.75, -1.5], [0.340, 0.4, 0.23, 3.1, 1.57, 1.5]]

# Robot Initialization
robot = get_robot('so100')
init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
init_gpos = lerobot_FK_5DOF(init_qpos[:5], robot=robot)
target_qpos, target_gpos = init_qpos.copy(), init_gpos.copy()
lock = threading.Lock()

# Key Mapping
key_map = {'w': (0, 1), 's': (0, -1), 'a': (1, 1), 'd': (1, -1), 'r': (2, 1), 'f': (2, -1),
           'q': (3, 1), 'e': (3, -1), 'g': (4, 1), 't': (4, -1), 'z': (5, 1), 'c': (5, -1)}
keys_pressed = {}

def on_press(key):
    try:
        k = key.char.lower()
        with lock:
            if k in key_map:
                keys_pressed[k] = key_map[k][1]
            elif k == "0":
                global target_qpos, target_gpos
                target_qpos, target_gpos = init_qpos.copy(), init_gpos.copy()
    except AttributeError:
        pass

def on_release(key):
    try:
        with lock:
            keys_pressed.pop(key.char.lower(), None)
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


try:
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()
            with lock:
                for k, direction in keys_pressed.items():
                    idx = key_map[k][0]
                    if idx == 5 :
                        if control_qlimit[0][idx] < target_qpos[idx] + JOINT_INCREMENT * direction < control_qlimit[1][idx]:
                            target_qpos[idx] += JOINT_INCREMENT * direction
                    
                    else:
                        if control_glimit[0][idx] < target_gpos[idx] + POSITION_INCREMENT * direction < control_glimit[1][idx]:
                            target_gpos[idx] += POSITION_INCREMENT * direction * (4 if idx in [3, 4] else 1)
            
            fd_qpos = mjdata.qpos[qpos_indices][:5]
            qpos_inv, ik_success, reg_gpos = lerobot_IK_5DOF(fd_qpos, target_gpos, robot=robot)
            
            if ik_success:
                target_qpos[:5] = qpos_inv[:5]
                mjdata.qpos[qpos_indices] = target_qpos
                mujoco.mj_step(mjmodel, mjdata)
                viewer.sync()
                print("target_gpos:", [f"{x:.3f}" for x in reg_gpos], "joint0:", target_qpos[0])
            
            time.sleep(max(0, mjmodel.opt.timestep - (time.time() - step_start)))
except KeyboardInterrupt:
    print("Simulation interrupted.")
finally:
    listener.stop()
    viewer.close()
