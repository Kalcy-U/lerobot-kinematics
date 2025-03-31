# code by LinCC111 Boxjod 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有

import numpy as np
import math
from math import sqrt as sqrt
from spatialmath import SE3
from lerobot_kinematics.ET import ET
from scipy.spatial.transform import Rotation as R
def create_so100():
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
    so100.qlim = [[-3.3, -0.2,     -1.8, -3.14158], 
                  [ 0.2,      3.14158,  1.8,  3.14158]]
    
    return so100

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
    T= SE3.Tx(0.0612) * SE3.Tz(0.0598) * SE3.Rz(qpos_data[0]) * T
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
    # 5DOF机械臂少一个自由度，这个自由度损失的结果是yaw和y绑定，yaw可以通过第0关节直接映射。
    yaw=math.atan2(y,x-0.0612)  # yaw的解析解
    T = SE3.Trans(x, y, z) * SE3.RPY(roll, pitch, yaw) 
    T = SE3.Rz(-yaw)*SE3.Tz(-0.0598) * SE3.Tx(-0.0612) * T 
    sol = robot.ikine_LM(
            Tep=T, 
            q0=q_now[1:5],
            ilimit=10,  # 10 iterations
            slimit=2,  # 1 is the limit
            tol=1e-3)
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
