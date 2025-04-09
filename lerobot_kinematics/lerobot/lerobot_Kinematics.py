# code by LinCC111 Boxjod 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有

import numpy as np
import math
from math import sqrt as sqrt
from spatialmath import SE3
from roboticstoolbox.robot.ET import ET
from scipy.spatial.transform import Rotation as R


class SO100Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._so100 = self._create_robot()
            self.BASE_TX=0.0612
            self.BASE_TZ=0.0598
    
    def _create_robot(self):
        # to joint 1
        # E1 = ET.tx(0.0612)
        # E2 = ET.tz(0.0598)
        # E3 = ET.Rz()
        
        # to joint 2
        E4 = ET.tx(0.02943)
        E5 = ET.tz(0.05504)
        E6 = ET.Ry()
        
        # to joint 3
        E7 = ET.tx(0.1127)
        E8 = ET.tz(-0.02798)
        E9 = ET.Ry()

        # to joint 4
        E10 = ET.tx(0.13504)
        E11 = ET.tz(0.00519)
        E12 = ET.Ry()
        
        # to joint 5
        E13 = ET.tx(0.0593)
        E14 = ET.tz(0.00996)
        E15 = ET.Rx()  
        
        # E17 = ET.tx(0.09538)
        # to gripper
        
        so100 = E4 * E5 * E6 * E7 * E8 * E9 * E10 * E11 * E12 * E13 * E14 * E15 #* E17  # E1 * E2 * E3 * 
        
        # Set joint limits
        so100.qlim = [[-3.5, -0.2,     -1.9, -3.14158], 
                    [ 0.2,      3.14158,  1.9,  3.14158]]
        return so100
    
    def __getattr__(self, name):
        return getattr(self._so100, name)
def create_so100():
    return SO100Singleton()

def get_robot(robot="so100"):
    
    if robot == "so100":
        return create_so100()
    else:
        print(f"Sorry, we don't support {robot} robot now")
        return None

def lerobot_FK(qpos_data, robot):
    if len(qpos_data) != len(robot.qlim[0]):
        raise Exception("The dimensions of qpose_data are not the same as the robot joint dimensions")
    # Get the end effector's homogeneous transformation matrix (T is an SE3 object)
    T = robot.fkine(qpos_data)
    
    r,p,y = T.rpy()
    X, Y, Z = T.t  
    return np.array([X, Y, Z, r, p, y])
    
def lerobot_FK_5DOF(qpos_data, robot):
    if len(qpos_data) != 5:
        raise Exception("The dimensions of qpose_data are not the same as the robot joint dimensions")
    T = robot.fkine(qpos_data[1:5])
    T= SE3.Tx(robot.BASE_TZ) * SE3.Tz(0.0598) * SE3.Rz(qpos_data[0]) * T
    r,p,y = T.rpy()
    X, Y, Z = T.t  
    return np.array([X, Y, Z, r, p, y])
"""
Performs inverse kinematics for a 5DOF robot arm.

Parameters:
- q_now: A list of the current joint angles of the robot arm.
- target_pose: A list representing the target pose (x, y, z, roll, pitch, yaw) of the robot arm's end effector.
- robot: robot object created by create_so100

Returns:
- qpose: The calculated joint angles to achieve the target pose.
- success: A boolean indicating whether the inverse kinematics operation was successful.
- actual_pose: The actual pose achieved by the calculated joint angles.
"""
def lerobot_IK_5DOF(q_now, target_pose, robot):
    if len(q_now) != 5:
        raise Exception("The dimensions of qpose_data are not 5")
    x, y, z, roll, pitch, yaw = list(target_pose)
    diff_tol=0.001
    # 5DOF机械臂少一个自由度，这个自由度损失的结果是yaw和y绑定，yaw可以通过第0关节直接映射。
    if ((x-robot.BASE_TZ)**2+y**2)<0.0009: # 处于基座半斤3cm范围内，空间可能不连续
        fk=lerobot_FK_5DOF(q_now, robot)
        diff = fk[0:5]-target_pose[0:5]
        if np.linalg.norm(diff)<0.16: 
            return q_now,True,[x, y, z, roll, pitch, yaw]
        else:
            
            diff_tol=1
    else:
        yaw=math.atan2(y,x-robot.BASE_TZ)  # yaw的解析解
        if yaw>math.pi/2:
            yaw=yaw-math.pi
        elif yaw<-math.pi/2:
            yaw=yaw+math.pi
        
    T = SE3.Trans(x, y, z) * SE3.RPY(roll, pitch, yaw) 
    T = SE3.Rz(-yaw)*SE3.Tz(-robot.BASE_TZ) * SE3.Tx(-robot.BASE_TZ) * T 
    sol = robot.ikine_LM(
            Tep=T, 
            q0=q_now[1:5],
            ilimit=10,  
            slimit=2, 
            tol=diff_tol)
    if sol.success:
        q = sol.q
        qpose=np.insert(q,0,yaw,axis=0)
        qpose = smooth_joint_motion(q_now, qpose, robot)
        
        return qpose,True,[x, y, z, roll, pitch,yaw ]
    else:
        print(f'IK fails')
        return -1 * np.ones(len(q_now)), False,[x, y, z, roll, pitch,yaw ]
    
def lerobot_IK(q_now, target_pose, robot):
    if len(q_now) != len(robot.qlim[0]):
        raise Exception("The dimensions of qpose_data are not the same as the robot joint dimensions")
    # R = SE3.RPY(target_pose[3:])
    # T = SE3(target_pose[:3]) * R
    
    x, y, z, roll, pitch, yaw = target_pose
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)  # 欧拉角的顺序是 XYZ
    R_mat = r.as_matrix()  # 获取旋转矩阵
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = [x, y, z]
    
    sol = robot.ikine_LM(
            Tep=T, 
            q0=q_now,
            ilimit=10,  # 10 iterations
            slimit=2,  # 1 is the limit
            tol=1e-3)  # tolerance for convergence
    
    if sol.success:
        # If IK solution is successful, 
        q = sol.q
        q = smooth_joint_motion(q_now, q, robot)
        # print(f'{q=}')
        return q, True
    else:
        # If the target position is unreachable, IK fails
        print(f'IK fails')
        return -1 * np.ones(len(q_now)), False
    
def smooth_joint_motion(q_now, q_new, robot):
    q_current = q_now
    max_joint_change = 0.1 
    
    for i in range(len(q_new)):
        delta = q_new[i] - q_current[i]
        if abs(delta) > max_joint_change:
            delta = np.sign(delta) * max_joint_change
        q_new[i] = q_current[i] + delta
    
    robot.q = q_new
    return q_new

def _lerobot_to_model(q):
    # Convert from LEROBOT dataset to ET MODEL coordinates
    c_qpos=q.copy()
    c_qpos[0:2]=-c_qpos[0:2]
    c_qpos=c_qpos*np.pi/180
    return c_qpos
def _model_to_lerobot(q):
    c_qpos=q.copy()
    c_qpos[0:2]=-c_qpos[0:2]
    c_qpos=c_qpos*180/np.pi
    return c_qpos
'''
q_now(lerobot dataset format)
target_pos(6dof gpos in meters+gripper in deg)
return qpos(in deg),ik_success(bool)
'''

def so100_IK(q_now, target_pos):
    
    robot = create_so100()
    c_qpos=_lerobot_to_model(q_now)
    q5,succ,target_g=lerobot_IK_5DOF(c_qpos[0:5], target_pos[0:6], robot)
    if succ:
        q5=_model_to_lerobot(q5)
        q_new = np.concatenate((q5,target_pos[6:])) # target_pos[6] is gripper in degree
        return q_new, True
    else:
        return -1 * np.ones(len(c_qpos)), False

'''
get qpos in lerobot format(in deg)
return gpos and gripper(in deg)
'''
def so100_FK(q_now):
    robot = create_so100()
    c_qpos=_lerobot_to_model(q_now)
    gpos=lerobot_FK_5DOF(c_qpos[0:5],robot)
    gpos=np.append(gpos,q_now[5]) #end effector
    return gpos # (xyzrpy,gripper)