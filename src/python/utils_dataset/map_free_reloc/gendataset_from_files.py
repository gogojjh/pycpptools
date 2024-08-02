"""
Author: Jianhao Jiao
Date: 2024-08-2
Description: This script generates camera data including RGB images, depth images, semantic images, and poses a ROS bag
Version: 1.0
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

"""Format of output map-free-reloc dataset
train or train or val/
	seq/
		frame_00000.jpg
		frame_00000.(mickey, zoe, zed).png
    poses.txt (format: image_path qw qx qy qz tx ty tz)
	intrinsics.txt (format: image_path fx fy cx cy width height)
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
		parser.add_argument('--config', type=str, default='/tmp/config.yaml', help='Path to configuration file')
		parser.add_argument('--in_dir', type=str, default='/tmp', help='Path to load data')
		parser.add_argument('--out_dir', type=str, default='/tmp', help='Path to save data')
		self.args = parser.parse_args()

		with open(self.args.config, 'r') as file:
			config = yaml.safe_load(file)

		self.poses = np.loadtxt(os.path.join(self.args.in_dir, 'poses.txt'))
		self.intrinsics = np.loadtxt(os.path.join(self.args.in_dir, 'intrinsics.txt'))
		self.kdtree = KDTree(self.poses[:, 1:4])			
		# self.keyframe_indices = [0, 200, 450, 950, 750, 8150, 9000, 9450, 10600, 11500, 17150]
		self.keyframe_indices =[200]
		self.img_width, self.img_height = int(self.intrinsics[0, 4]), int(self.intrinsics[0, 5])
			
	def setup_directories(self):
		base_path = f'{self.args.out_dir}'
		paths = [base_path]
		for path in paths:
			os.makedirs(path, exist_ok=True)
		self.base_path = base_path

	def run(self):
		radius = 15.0
		for scene_id, indice in enumerate(self.keyframe_indices):
			result = self.kdtree.query_ball_point(self.poses[indice, 1:4], r=radius)
			result.sort()
			ref_intrinsics = np.array([self.intrinsics[indice, 0], 0, self.intrinsics[indice, 2], 
							  		   0, self.intrinsics[indice, 1], self.intrinsics[indice, 3], 
									   0, 0, 1]).reshape(3, 3)
			trans, quat = self.poses[indice, 1:4], self.poses[indice, 4:]
			T_w_ref = convert_vec_to_matrix(trans, quat, 'xyzw')
	
			os.makedirs(os.path.join(self.args.out_dir, f's{scene_id:05d}'), exist_ok=True)
			os.makedirs(os.path.join(self.args.out_dir, f's{scene_id:05d}', 'seq0'), exist_ok=True)
			os.makedirs(os.path.join(self.args.out_dir, f's{scene_id:05d}', 'seq1'), exist_ok=True)
			all_intrinsics, all_poses = np.empty((0, 7), dtype=object), np.empty((0, 8), dtype=object)
			vec = np.empty((1, 7), dtype=object)
			vec[0, 0], vec[0, 1:] = f'seq0/frame_{0:05d}.jpg', self.intrinsics[indice, :]
			all_intrinsics = np.vstack((all_intrinsics, vec))
			vec = np.empty((1, 8), dtype=object)
			vec[0, 0], vec[0, 1:5], vec[0, 5:] = f'seq0/frame_{0:05d}.jpg', np.array([1, 0, 0, 0]), np.array([0, 0, 0])
			all_poses = np.vstack((all_poses, vec))

			rgb_img_path = os.path.join(self.args.in_dir, 'seq', f'{indice:06d}.color.jpg')
			depth_img_path = os.path.join(self.args.in_dir, 'seq', f'{indice:06d}.depth.png')
			new_rgb_img_path = os.path.join(self.args.out_dir, f's{scene_id:05d}', 'seq0', f'frame_{0:05d}.jpg')
			new_depth_img_path = os.path.join(self.args.out_dir, f's{scene_id:05d}', 'seq0', f'frame_{0:05d}.zed.png')			
			os.system(f'cp {rgb_img_path} {new_rgb_img_path}')
			os.system(f'cp {depth_img_path} {new_depth_img_path}')
			new_img_id = 0
			for id in result[::10]:
				target_intrinsics = np.array([self.intrinsics[id, 0], 0, self.intrinsics[id, 2], 
											  0, self.intrinsics[id, 1], self.intrinsics[id, 3], 
											  0, 0, 1]).reshape(3, 3)
				T_w_target = convert_vec_to_matrix(self.poses[id, 1:4], self.poses[id, 4:], 'xyzw')
				T_target_ref = np.linalg.inv(T_w_target) @ T_w_ref

				fake_depth_map = np.zeros((self.img_height, self.img_width))
				fake_depth_map.fill(7.0)
				depth_points = depth_image_to_point_cloud(fake_depth_map, ref_intrinsics, [self.img_width, self.img_height])
				transformed_depth_points = transform_point_cloud(depth_points, T_target_ref)
				proj_depth_map = project_point_cloud(transformed_depth_points, target_intrinsics, [self.img_width, self.img_height])
				valid_mask = proj_depth_map > 0
				overlap = np.sum(valid_mask) / (self.img_width * self.img_height)
				if overlap > 0.4:
					trans, quat = convert_matrix_to_vec(T_target_ref, 'xyzw')
					vec = np.empty((1, 7), dtype=object)
					vec[0, 0], vec[0, 1:] = f'seq1/frame_{new_img_id:05d}.jpg', self.intrinsics[id, :]
					all_intrinsics = np.vstack((all_intrinsics, vec))
					vec = np.empty((1, 8), dtype=object)
					vec[0, 0], vec[0, 1:5], vec[0, 5:] = f'seq1/frame_{new_img_id:05d}.jpg', np.roll(quat, 1), trans
					all_poses = np.vstack((all_poses, vec))

					rgb_img_path = os.path.join(self.args.in_dir, 'seq', f'{id:06d}.color.jpg')
					depth_img_path = os.path.join(self.args.in_dir, 'seq', f'{id:06d}.depth.png')
					new_rgb_img_path = os.path.join(self.args.out_dir, f's{scene_id:05d}', 'seq1', f'frame_{new_img_id:05d}.jpg')
					new_depth_img_path = os.path.join(self.args.out_dir, f's{scene_id:05d}', 'seq1', f'frame_{new_img_id:05d}.zed.png')			
					os.system(f'cp {rgb_img_path} {new_rgb_img_path}')
					os.system(f'cp {depth_img_path} {new_depth_img_path}')
					new_img_id += 1
				print(overlap)
			np.savetxt(os.path.join(self.base_path, f's{scene_id:05d}', 'intrinsics.txt'), all_intrinsics, fmt='%s %.9f %.9f %.9f %.9f %.9f %.9f')
			np.savetxt(os.path.join(self.base_path, f's{scene_id:05d}', 'poses.txt'), all_poses, fmt='%s %.9f %.9f %.9f %.9f %.9f %.9f %.9f')
			# input()

if __name__ == '__main__':
	data_generator = DataGenerator()
	data_generator.setup_directories() 
	data_generator.run()

