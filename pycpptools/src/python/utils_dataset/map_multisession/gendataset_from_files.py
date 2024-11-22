"""
Author: Jianhao Jiao
Date: 2024-11-22
Description: This script generates camera data including RGB images, depth images, semantic images, and poses a ROS bag
Version: 1.0
"""

"""
Usage: python gendataset_from_files.py --in_dir out_general --out_dir map_multisession_eval --num_split 2 --scene_id 0 --start_indice 0
"""

"""Format of input dataset
out_general/
    seq/
        000000.color.png
        000000.depth.png (mm)
		000000.semantic.png
    poses.txt (TUM format: timestamp tx ty tz qx qy qz qw)
    intrinsics.txt (format: fx fy cx cy width height)
"""

"""Format of output map-multisession mapping dataset
train or train or val/
	s00000/
		seq0/
			frame_00000.jpg
			frame_00000.(mickey, zoe, zed).png
			...
		seq1/
			frame_00000.jpg
			frame_00000.(mickey, zoe, zed).png
			...
		...
		seqN/
			frame_00000.jpg
			frame_00000.(mickey, zoe, zed).png
	poses.txt (format: image_path qw qx qy qz tx ty tz) - absolute poses in the world
	intrinsics.txt (format: image_path fx fy cx cy width height)
    poses_seq0.txt (format: image_path qw qx qy qz tx ty tz) - relative poses
	poses_seq1.txt (format: image_path qw qx qy qz tx ty tz) - relative poses
	...
	poses_seqN.txt (format: image_path qw qx qy qz tx ty tz) - relative poses
"""

import os
import sys
import yaml
import numpy as np
import argparse
from scipy.spatial import KDTree
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from utils_math.tools_eigen import convert_matrix_to_vec, convert_vec_to_matrix

def depth_image_to_point_cloud(depth_image, intrinsics, image_shape):
    """
    Convert a depth image to a point cloud.

    Parameters:
    depth_image (numpy.ndarray): The depth image.
    intrinsics (numpy.ndarray): The camera intrinsic matrix.

    Returns:
    numpy.ndarray: The point cloud as an (N, 3) array.
    """
    w, h = image_shape
    i, j = np.indices((h, w))
    z = depth_image
    x = (j - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (i - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    return points

def transform_point_cloud(points, transformation_matrix):
    """
    Apply a transformation to a point cloud.

    Parameters:
    points (numpy.ndarray): The point cloud as an (N, 3) array.
    transformation_matrix (numpy.ndarray): The 4x4 transformation matrix.

    Returns:
    numpy.ndarray: The transformed point cloud as an (N, 3) array.
    """
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed = points_homogeneous @ transformation_matrix.T
    return points_transformed[:, :3]

def project_point_cloud(points, intrinsics, image_shape):
    """
    Project a point cloud onto an image plane.

    Parameters:
    points (numpy.ndarray): The point cloud as an (N, 3) array.
    intrinsics (numpy.ndarray): The camera intrinsic matrix.
    image_shape (tuple): The shape of the output image (height, width).

    Returns:
    numpy.ndarray: The projected depth image.
    """
    w, h = image_shape
    z = points[:, 2]
    x = (points[:, 0] * intrinsics[0, 0] / z + intrinsics[0, 2]).astype(np.int32)
    y = (points[:, 1] * intrinsics[1, 1] / z + intrinsics[1, 2]).astype(np.int32)

    depth_image = np.zeros((h, w))
    valid_mask = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (z > 0)
    depth_image[y[valid_mask], x[valid_mask]] = z[valid_mask]
    return depth_image	

def draw_images(image0, image1):
	# Create a figure and a set of subplots
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))
	# Display the first image with a colorbar
	im1 = axes[0].imshow(image0, cmap='viridis')
	axes[0].axis('off')
	fig.colorbar(im1, ax=axes[0])
	# Display the second image with a colorbar
	im2 = axes[1].imshow(image1, cmap='viridis')
	axes[1].axis('off')
	fig.colorbar(im2, ax=axes[1])	
	plt.show()

class DataGenerator:
	def __init__(self):
		# Initialize argument parser and add arguments
		parser = argparse.ArgumentParser(description="Camera data collector for synchronized RGBD images and poses.")
		parser.add_argument('--in_dir', type=str, default='/tmp', help='Path to load data')
		parser.add_argument('--out_dir', type=str, default='/tmp', help='Path to save data')
		parser.add_argument('--num_split', type=int, default=2, help='Number of splits')
		parser.add_argument('--scene_id', type=int, default=0, help='Scene ID')
		parser.add_argument('--start_indice', type=int, default=0, help='Start indice')
		parser.add_argument('--step', type=int, default=1, help='Step')
		print('Usage: python gendataset_from_files.py --config config_matterport3d.yaml --in_dir out_general --out_dir map_multisession_eval --scene_id 0 --start_indice 0')
		self.args = parser.parse_args()

		self.poses = np.loadtxt(os.path.join(self.args.in_dir, 'poses.txt'))
		self.intrinsics = np.loadtxt(os.path.join(self.args.in_dir, 'intrinsics.txt'))
		self.start_indice = self.args.start_indice
			
	def setup_directories(self):
		base_path = os.path.join(self.args.out_dir, f's{self.args.scene_id:05d}')
		paths = [base_path]
		for i in range(self.args.num_split):
			paths.append(os.path.join(base_path, f'out_map{i}'))
			paths.append(os.path.join(base_path, f'out_map{i}/seq'))
		for path in paths:
			os.makedirs(path, exist_ok=True)
		self.base_path = base_path

	def run(self):
		total_len = len(self.poses) 
		N = self.args.num_split
		path_segments = [
			(int(max(0, i * total_len / N)), int(min(total_len, (i + 1) * total_len / N)))
			for i in range(N)
		]
		print(f'Total length of segment: {total_len}, split {N} segments')
		print('Segments: ', path_segments)
		for seg_id, (start_ind, end_ind) in enumerate(path_segments):
			print(f'Processing segment {seg_id} from {start_ind} to {end_ind}')
			seg_intrinsics = np.empty((0, 6), dtype=object)
			seg_poses_abs_gt = np.empty((0, 8), dtype=object)
			seg_poses_abs_noise = np.empty((0, 8), dtype=object)
			edges = np.empty((0, 3), dtype=object)
			# time, tsl, quat = self.poses[start_ind, 0], self.poses[start_ind, 1:4], self.poses[start_ind, 4:]
			# T_w2c0 = convert_vec_to_matrix(tsl, quat, 'xyzw')
			for ind in range(start_ind, end_ind, self.args.step):
				cur_ind = int((ind - start_ind) / self.args.step)
				##### Intrinsics
				vec = np.empty((1, 6), dtype=object)
				vec[0, 0:4], vec[0, 4], vec[0, 5] = \
					self.intrinsics[ind, :4], int(self.intrinsics[ind, 4]), int(self.intrinsics[ind, 5])
				seg_intrinsics = np.vstack((seg_intrinsics, vec))
				##### Poses in the absolute world frame of all frames (gt)
				vec = np.empty((1, 8), dtype=object)
				time, tsl, quat = self.poses[ind, 0], self.poses[ind, 1:4], self.poses[ind, 4:]
				vec[0, 0], vec[0, 1:4], vec[0, 4:] = time, tsl, quat
				seg_poses_abs_gt = np.vstack((seg_poses_abs_gt, vec))
				##### Poses in the absolute world frame of all frames (noise)
				vec = np.empty((1, 8), dtype=object)
				time, tsl, quat = self.poses[ind, 0], self.poses[ind, 1:4], self.poses[ind, 4:]
				vec[0, 0], vec[0, 1:4], vec[0, 4:] = time, tsl, quat
				seg_poses_abs_noise = np.vstack((seg_poses_abs_noise, vec))				
				##### Poses in the relative world frame of segmented frames
				# time, tsl, quat = self.poses[ind, 0], self.poses[ind, 1:4], self.poses[ind, 4:]
				# T_w2ct = convert_vec_to_matrix(tsl, quat, 'xyzw')
				# T_w2c_0t = np.linalg.inv(T_w2c0) @ T_w2ct
				# tsl_0t, quat_0t = convert_matrix_to_vec(T_w2c_0t, 'xyzw')
				# vec = np.empty((1, 8), dtype=object)
				# vec[0, 0], vec[0, 1:4], vec[0, 4:] = time, tsl_0t, quat_0t
				# seg_poses_rel = np.vstack((seg_poses_rel, vec))
				##### Edges
				if ind > start_ind:
					dis = np.linalg.norm(self.poses[ind - 1, 1:4] - self.poses[ind, 1:4])
					edges = np.vstack((edges, np.array([cur_ind - 1, cur_ind, dis])))
				##### Save images from segmented frames
				rgb_img_path = os.path.join(self.args.in_dir, 'seq', f'{ind:06d}.color.jpg')
				depth_img_path = os.path.join(self.args.in_dir, 'seq', f'{ind:06d}.depth.png')
				sem_img_path = os.path.join(self.args.in_dir, 'seq', f'{ind:06d}.semantic.png')
				new_rgb_img_path = os.path.join(self.base_path, f'out_map{seg_id}/seq', f'{cur_ind:06d}.color.jpg')
				new_depth_img_path = os.path.join(self.base_path, f'out_map{seg_id}/seq', f'{cur_ind:06d}.depth.png')			
				new_sem_img_path = os.path.join(self.base_path, f'out_map{seg_id}/seq', f'{cur_ind:06d}.semantic.png')
				os.system(f'cp {rgb_img_path} {new_rgb_img_path}')
				os.system(f'cp {depth_img_path} {new_depth_img_path}')
				os.system(f'cp {sem_img_path} {new_sem_img_path}')
			np.savetxt(os.path.join(self.base_path, f'out_map{seg_id}/intrinsics.txt'), seg_intrinsics, fmt='%.9f %.9f %.9f %.9f %d %d')
			np.savetxt(os.path.join(self.base_path, f'out_map{seg_id}/poses.txt'), seg_poses_abs_noise, fmt='%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f')
			np.savetxt(os.path.join(self.base_path, f'out_map{seg_id}/poses_gt.txt'), seg_poses_abs_gt, fmt='%.9f %.9f %.9f %.9f %.9f %.9f %.9f %9f')
			np.savetxt(os.path.join(self.base_path, f'out_map{seg_id}/edge_list.txt'), edges, fmt='%d %d %.9f')
		print('Finish generating dataset')
		# input()

if __name__ == '__main__':
	data_generator = DataGenerator()
	data_generator.setup_directories() 
	data_generator.run()

