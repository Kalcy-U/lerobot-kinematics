'''
follow action sequence from lerobot dataset
test FK and IK
'''


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

# Robot Initialization
robot = get_robot('so100')

lock = threading.Lock()
qpos_list= load_dataset_qpos()
count_bias_a=0
count_bias_b=0
try:
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        for qpos in qpos_list:
            #lerobot space to model space
            qpos = to_rad(qpos)
            qpos[0:2]=-qpos[0:2] 
            
            mjdata.qpos[0:6]=qpos
            mujoco.mj_step(mjmodel, mjdata)
            viewer.sync()
            time.sleep(0.05)
            position = lerobot_FK_5DOF(qpos[0:5], robot=robot)
            qpos_inv, ik_success, reg_pos = lerobot_IK_5DOF(qpos[0:5], position, robot=robot)
           
            print("       gpos:", position)
            if ik_success:
                qpos_diff = qpos[0:5] - qpos_inv[0:5]
                if np.sum(qpos_diff**2) > 2:
                    print("origin qpos:", [f"{x:.4f}" for x in qpos[0:5]])
                    print("IK     qpos:", [f"{x:.4f}" for x in qpos_inv[0:5]])
                    print("IK     diff:", [f"{x:.4f}" for x in qpos_diff])
                    count_bias_b+=1
            else:
                print("origin qpos:", [f"{x:.4f}" for x in qpos[0:5]])
                count_bias_a+=1
            print("-" * 50)

except KeyboardInterrupt:
    print("Simulation interrupted.")
finally:
    viewer.close()
    
if count_bias_a>0 or count_bias_b>0:
    print(f'Warning report:\nLarge bias:{count_bias_a}, IK failed:{count_bias_b}')
else:
    print("success")