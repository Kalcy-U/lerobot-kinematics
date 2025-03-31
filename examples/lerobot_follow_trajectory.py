import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import math
from pynput import keyboard
from lerobot_kinematics import lerobot_IK_5DOF, lerobot_FK_5DOF, get_robot
import pandas as pd

def load_dataset_qpos(path='episode_000000.parquet'):
    df = pd.read_parquet(path)
    return df['observation.state'].to_list()

to_rad = lambda x: x *math.pi/180

np.set_printoptions(linewidth=200)
os.environ["MUJOCO_GL"] = "egl"

# Load Model
xml_path = "./examples/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
mjdata = mujoco.MjData(mjmodel)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]])

# Robot Initialization
robot = get_robot('so100')

init_qpos = to_rad(np.array([ 0.26367188 ,184.7461  ,   100.634766 ,   99.66797    , -2.109375,   9.083263 ]))
print(init_qpos)
lock = threading.Lock()

qpos_list= load_dataset_qpos()

try:
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        for qpos in qpos_list:
            qpos = to_rad(qpos)
            qpos[0:2]=-qpos[0:2]
            mjdata.qpos[0:6]=qpos
            mujoco.mj_step(mjmodel, mjdata)
            viewer.sync()
            time.sleep(0.05)
        
except KeyboardInterrupt:
    print("Simulation interrupted.")
finally:
    viewer.close()
