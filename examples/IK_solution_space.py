import numpy as np
import matplotlib.pyplot as plt
from lerobot_kinematics import lerobot_IK_5DOF,get_robot
import math
from scipy.spatial.transform import Rotation as R

robot=get_robot('so100')
x_range = np.linspace(0., 0.5, 10)  
y_range = np.linspace(-0.5, 0.5, 10)
z_range = np.linspace(0., 0.4, 10)  

ik_solutions = []
failed_points = []
init_qpos = [0.0, -3.14, 3.14, 0.0, -1.57]

for x in x_range:
    for y in y_range:
        for z in z_range:
            qpos_inv, ik_success, reg_gpos = lerobot_IK_5DOF(init_qpos, [x,y,z,0,0,0],robot)
            if ik_success:
                # target_qpos = qpos_inv[:5]
                ik_solutions.append([x, y, z])
            else:
                failed_points.append([x, y, z])

ik_solutions = np.array(ik_solutions)
failed_points = np.array(failed_points)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

origin = np.array([[0, 0, 0]])
axes = np.array([[0.1, 0, 0],  # X 
                 [0, 0.1, 0],  # Y 
                 [0, 0, 0.1]]) # Z 

ax.quiver(*origin.T, *axes.T, color=['r', 'g', 'b'], length=0.5, linewidth=2)

if len(ik_solutions) > 0:
    ax.scatter(ik_solutions[:, 0], ik_solutions[:, 1], ik_solutions[:, 2], 
               c='b', marker='o', label="IK Success (Reachable)")

if len(failed_points) > 0:
    ax.scatter(failed_points[:, 0], failed_points[:, 1], failed_points[:, 2], 
               c='r', marker='x', label="IK Fail (Unreachable)")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("SO100 IK Solution Space")
ax.legend()
plt.show()
