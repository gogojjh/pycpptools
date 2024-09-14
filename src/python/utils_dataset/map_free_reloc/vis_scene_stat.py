"""
Usage:

python vis_scene_stat.py --dataset /Rocket_ssd/dataset/data_topo_loc/matterport3d/map_free_eval/test/ --dataset_name Matterport3d --sample_rate 1
python vis_scene_stat.py --dataset /Rocket_ssd/dataset/data_topo_loc/ucl_campus/map_free_eval/test/ --dataset_name UCLCampus --sample_rate 1
python vis_scene_stat.py --dataset /Rocket_ssd/dataset/data_topo_loc/hkustgz_campus/map_free_eval/test/ --dataset_name HKUSTGZCampus --sample_rate 1
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../utils_visualization"))
from tools_color_setting import PALLETE

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../utils_math"))
from tools_eigen import convert_vec_to_matrix, convert_matrix_to_vec


import numpy as np
import argparse

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    import matplotlib
    matplotlib.use('Agg')  # set the backend before importing pyplot

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import pylab, colors
from colorama import init, Fore

# init(autoreset=True)
plt.rcParams['font.family'] = 'DejaVu Serif'
params = {'axes.titlesize': 12, 'legend.fontsize': 8, 'legend.numpoints': 1}
pylab.rcParams.update(params)

def plot_cameras_arrow(poses, sample_rate, title='Camera Observations', ax=None):
    bounds = np.max(poses[:, :3], axis=0) - np.min(poses[:, :3], axis=0)
    max_bound = np.max(bounds) / 2
    arrow_length = max_bound / 10
    head_width = max_bound / 15
    head_length = max_bound / 15
    bound_limit = max_bound / 5

    # based on camera cooordinate: x-right, y-down, z-forward   
    if ax is None:
        fig = plt.figure(figure=(5, 5))
        ax = fig.add_subplot(111, aspect='equal', xlabel='X [m]', ylabel='Z [m]')

    points = np.empty((0, 3))
    T_ini = np.eye(4)
    for i, pose in enumerate(poses[::sample_rate, :]):
        trans, quat = pose[:3], pose[3:]
        T = convert_vec_to_matrix(trans, quat, mode='xyzw')
        T_wcam_cam = np.dot(np.linalg.inv(T_ini), T)
        points = np.vstack((points, T_wcam_cam[:3, 3]))

        vector_start = np.dot(T_wcam_cam[:3, :3], np.array([0, 0, 0]).reshape(3, 1)) + T_wcam_cam[:3, 3].reshape(3, 1) 
        vector_end = np.dot(T_wcam_cam[:3, :3], np.array([0, 0, arrow_length]).reshape(3, 1)) + T_wcam_cam[:3, 3].reshape(3, 1)
        dir = vector_end - vector_start
        if i == 0:
            ax.arrow(vector_start[0][0], vector_start[2][0], dir[0][0], dir[2][0], head_width=head_width*2.0, head_length=head_length*1.5, width=0.020, fc=PALLETE[0], ec=PALLETE[0], zorder=100)
        else:
            ax.arrow(vector_start[0][0], vector_start[2][0], dir[0][0], dir[2][0], head_width=head_width, head_length=head_length, width=0.015, fc=PALLETE[1], ec=PALLETE[1])
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])
    ax.set_xlim(min_x - bound_limit, max_x + bound_limit)
    ax.set_ylim(min_z - bound_limit, max_z + bound_limit)
    ax.set_aspect('equal', 'box')
    ax.grid(ls='--', color='0.7')
    plt.title(title)

def plot_cameras_3daxis(poses, sample_rate, title='Camera Observations', ax=None, color=None):
    # based on camera cooordinate: x-right, y-down, z-forward   
    if ax is None:
        fig = plt.figure(figure=(5, 5))
        ax = fig.add_subplot(111, aspect='equal', xlabel='X [m]', ylabel='Y [m]', zlabel='Z [m]', projection='3d')

    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    points = np.empty((0, 3))

    T_ini = np.eye(4)
    for i, pose in enumerate(poses[::sample_rate, :]):
        trans, quat = pose[:3], pose[3:]
        T = convert_vec_to_matrix(trans, quat, mode='xyzw')
        T_wcam_cam = np.dot(np.linalg.inv(T_ini), T)
        points = np.vstack((points, T_wcam_cam[:3, 3]))

        origin = T_wcam_cam[:3, 3]
        rotation = T_wcam_cam[:3, :3]

        new_x_axis = rotation @ x_axis
        new_y_axis = rotation @ y_axis
        new_z_axis = rotation @ z_axis
        ax.quiver(origin[0], origin[1], origin[2], new_x_axis[0], new_x_axis[1], new_x_axis[2], length=0.4, normalize=True, color=[1, 0, 0], arrow_length_ratio=0.3)
        ax.quiver(origin[0], origin[1], origin[2], new_y_axis[0], new_y_axis[1], new_y_axis[2], length=0.4, normalize=True, color=[0, 1, 0], arrow_length_ratio=0.3)
        ax.quiver(origin[0], origin[1], origin[2], new_z_axis[0], new_z_axis[1], new_z_axis[2], length=0.4, normalize=True, color=[0, 0, 1], arrow_length_ratio=0.3)

    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])
    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)
    ax.set_zlim(min_z - 1, max_z + 1)
    ax.view_init(elev=0, azim=-90, roll=0, vertical_axis='z')
    ax.set_aspect('equal')
    ax.grid(ls='--', color='0.7')
    ax.set_title(title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="path to dataset")
    parser.add_argument("--dataset_name", help="Matterport3d, UCLCampus, HKUSTGZCampus")
    parser.add_argument("--sample_rate", type=int, default=1, help="sample rate for visualization")
    args = parser.parse_args()

    # Load dataset
    all_scenes = os.listdir(args.dataset)
    all_scenes.sort()

    num_ref = 0
    num_query = 0
    for scene in all_scenes:
        scene_path = os.path.join(args.dataset, scene)
        if not os.path.isdir(scene_path): continue
        print(Fore.GREEN + 'Processing {} at {} ...'.format(scene, scene_path))

        str_poses = np.genfromtxt(os.path.join(scene_path, 'poses.txt'), dtype=None, encoding=None)
        poses = np.zeros((len(str_poses), 7))
        for i, str_pose in enumerate(str_poses):
            data = []
            for j in range(len(str_pose) - 1):
                data.append(float(str_pose[j+1]))
            data = np.array(data)
            quat, trans = np.roll(data[:4], -1), data[4:]
            T_inv = np.linalg.inv(convert_vec_to_matrix(trans, quat))
            trans, quat = convert_matrix_to_vec(T_inv)
            poses[i, :3] = trans
            poses[i, 3:] = quat
        title = '{}-{}-{} frames'.format(args.dataset_name, scene, len(poses)-1)
        plot_cameras_arrow(poses, title=title, sample_rate=args.sample_rate)
        # plt.show()
        plt.savefig(os.path.join(args.dataset, '../scene_stat', '{}_poses.pdf'.format(scene)))
        plt.close()
        # break

        num_ref += 1
        num_query += len(poses) - 1

    print(Fore.GREEN + 'Total {} (reference) scenes, {} query images'.format(num_ref, num_query))
