#!/usr/bin/python3

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../utils_math'))

import argparse
import numpy as np

from tools_eigen import convert_matrix_to_vec, convert_vec_to_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--in_odom_path', type=str, help='/tmp/input_odom.txt')
parser.add_argument('--out_odom_path', type=str, help='/tmp/output_odom.txt')
parser.add_argument('--format', type=str, default='TUM', help='TUM, KITTI')
args = parser.parse_args()
print("Arguments:\n{}".format('\n'.join(
	['-{}: {}'.format(k, v) for k, v in args.__dict__.items()])))

###############################
# Matterport3d
trans = np.array([0, 0, 0])
quat = np.array([-0.500, 0.500, -0.500, 0.500])
T_sensor_cam = convert_vec_to_matrix(trans, quat, mode='xyzw')
# T_offset = T_sensor_cam
###############################

###############################
# Anymal
# trans = np.array([-0.377, -0.001, -0.482])
# quat = np.array([-0.013, -0.141, -0.008, 0.990])
# T_livox_base = convert_vec_to_matrix(trans, quat, mode='xyzw')
# T_offset = T_livox_base

# trans = np.array([0.004, 0.018, -0.024])
# quat = np.array([-0.509, 0.453, -0.526, 0.509])
# T_livox_cam = convert_vec_to_matrix(trans, quat, mode='xyzw')
# T_offset = T_livox_cam
###############################

###############################
T_offset = np.eye(4)
###############################

def transform_odom(args):
	poses = np.loadtxt(args.in_odom_path)
	new_poses = np.zeros_like(poses)

	trans, quat = poses[0][1:4], poses[0][4:8]
	T0 = convert_vec_to_matrix(trans, quat, mode='xyzw')
	for i, pose in enumerate(poses):
		time, trans, quat = pose[0], pose[1:4], pose[4:8]
		T_world_sensor_sensor = convert_vec_to_matrix(trans, quat, mode='xyzw')
		T = T_offset @ np.linalg.inv(T0) @ T_world_sensor_sensor @ np.linalg.inv(T_offset)
		new_poses[i, 0] = time
		new_poses[i, 1:4], new_poses[i, 4:8] = convert_matrix_to_vec(T, mode='xyzw')
	np.savetxt(args.out_odom_path, new_poses, '%.9f')

if __name__ == '__main__':
	if args.format == 'TUM':
		transform_odom(args)
	else:
		print('Unsupported format: {}'.format(args.format))
		exit(1)

