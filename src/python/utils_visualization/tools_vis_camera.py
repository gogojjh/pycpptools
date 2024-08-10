import argparse
import numpy as np
import matplotlib.pyplot as plt

from pycpptools.src.python.utils_math.tools_eigen import convert_vec_to_matrix

def plot_cameras(poses, sample_rate, title='Camera Observations', ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    T_ini = np.eye(4)
    for i, pose in enumerate(poses[::sample_rate, :]):
        trans, quat = pose[:3], pose[3:]
        T = convert_vec_to_matrix(trans, quat, mode='xyzw')
        T_wcam_cam = np.linalg.inv(T_ini) @ T

        vector_start = T_wcam_cam[:3, :3] @ np.array([0, 0, 0]).reshape(3, 1) + T_wcam_cam[:3, 3].reshape(3, 1) 
        vector_end = T_wcam_cam[:3, :3] @ np.array([0, 0, 2]).reshape(3, 1) + T_wcam_cam[:3, 3].reshape(3, 1)
        dir = vector_end - vector_start
        color = plt.cm.viridis(i / len(poses[::sample_rate, :]))  # Use colormap to show pose in history
        ax.quiver(vector_start[0], vector_start[1], dir[0], dir[1], color=color)
    
    min_x, max_x = np.min(poses[:, 0]), np.max(poses[:, 0])
    min_y, max_y = np.min(poses[:, 1]), np.max(poses[:, 1])
    plt.xlim(min_x - 2, max_x + 2)
    plt.ylim(min_y - 2, max_y + 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    plt.title(title)

def plot_connected_cameras(poses, edge_list, title='Connected Camera Observations', ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    T_ini = np.eye(4)
    # Plot cameras
    for i, pose in enumerate(poses):
        trans, quat = pose[:3], pose[3:]
        T = convert_vec_to_matrix(trans, quat, mode='xyzw')
        T_wcam_cam = np.linalg.inv(T_ini) @ T

        vector_start = T_wcam_cam[:3, :3] @ np.array([0, 0, 0]).reshape(3, 1) + T_wcam_cam[:3, 3].reshape(3, 1) 
        vector_end = T_wcam_cam[:3, :3] @ np.array([0, 0, 2]).reshape(3, 1) + T_wcam_cam[:3, 3].reshape(3, 1)
        dir = vector_end - vector_start
        color = plt.cm.viridis(i / len(poses))  # Use colormap to show pose in history
        ax.quiver(vector_start[0], vector_start[1], dir[0], dir[1], color=color)
    # Plot connection
    for edge in edge_list:
        node_id0, node_id1 = int(edge[0]), int(edge[1])
        trans, quat = poses[node_id0, :3], poses[node_id0, 3:]
        T = convert_vec_to_matrix(trans, quat, mode='xyzw')
        T0 = np.linalg.inv(T_ini) @ T
        trans, quat = poses[node_id1, :3], poses[node_id1, 3:]
        T = convert_vec_to_matrix(trans, quat, mode='xyzw')
        T1 = np.linalg.inv(T_ini) @ T
        ax.plot([T0[0, 3], T1[0, 3]], [T0[1, 3], T1[1, 3]], '-', color='k', linewidth=0.5)

    min_x, max_x = np.min(poses[:, 0]), np.max(poses[:, 0])
    min_y, max_y = np.min(poses[:, 1]), np.max(poses[:, 1])
    plt.xlim(min_x - 2, max_x + 2)
    plt.ylim(min_y - 2, max_y + 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    plt.title(title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot camera poses.')
    parser.add_argument('--path_pose', type=str, required=True, help='Path to the poses file')
    parser.add_argument('--sample_rate', type=int, default=10, help='Sample rate for the poses')
    args = parser.parse_args()

    poses = np.loadtxt(args.path_pose)[:, 1:] # tx, ty, tz, qx, qy, qz, qw
    plot_cameras(poses, args.sample_rate)

    edge_list = np.empty((0, 3), dtype=np.float32)
    edge_list = np.vstack((edge_list, np.array([0, 3, 1])))
    edge_list = np.vstack((edge_list, np.array([3, 6, 1])))
    edge_list = np.vstack((edge_list, np.array([0, 6, 1])))
    plot_connected_cameras(poses, edge_list)
    
    plt.show()
