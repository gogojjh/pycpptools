import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pycpptools.src.python.utils_math.tools_eigen import convert_vec_to_matrix, convert_matrix_to_vec

def plot_cameras(poses, sample_rate):
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
    plt.title('Camera Observations')
    plt.show()

def main(args):
    # Load poses from file
    poses = np.loadtxt(args.path_pose)[:, 1:] # tx, ty, tz, qx, qy, qz, qw
    plot_cameras(poses, args.sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot camera poses.')
    parser.add_argument('--path_pose', type=str, required=True, help='Path to the poses file')
    parser.add_argument('--sample_rate', type=int, default=10, help='Sample rate for the poses')
    args = parser.parse_args()
    main(args)