"""
Author: Jianhao Jiao
Date: 2024-11-22
Description: This script generates camera data including RGB images, depth images, semantic images, and poses a ROS bag
Version: 1.0
"""

"""
Usage: python gendataset_from_files.py \
	--in_dir out_general \
	--out_dir map_multisession_eval/place_name \
	--num_split 2 \
	--scene_id 0 \
	--start_indice 0 \
	--step 50 \
	--offset 0
"""

"""Format of input dataset
out_general/
    seq/
        000000.color.png
        000000.depth.png (mm)
		000000.semantic.png
    poses_abs_odom.txt    (TUM format: timestamp tx ty tz qx qy qz qw)
	poses_abs_gt.txt (TUM format: timestamp tx ty tz qx qy qz qw)
    intrinsics.txt (format: fx fy cx cy width height)
"""

"""Format of output map-multisession mapping dataset
train or train or val/
	s00000/
		seq0/
			frame_00000.jpg
			frame_00000.(mickey, zoe, zed).png
			poses.txt (format: image_name qw qx qy qz tx ty tz) - poses odometry in the relative world frame
			poses_rel_gt.txt (format: image_name qw qx qy qz tx ty tz) - poses gt in the relative world frame
			poses_abs_gt.txt (format: image_name qw qx qy qz tx ty tz) - poses gt in the absoluted world frame
			intrinsics.txt (format: image_name fx fy cx cy width height)
			timestamps.txt (format: image_name timestamp)
			edge_list.txt (format: id0 id1 weight)
			database_descriptors.txt (format: image_name descriptor)
			...
		seq1/
			frame_00000.jpg
			frame_00000.(mickey, zoe, zed).png
			...
		...
		seqN/
			frame_00000.jpg
			frame_00000.(mickey, zoe, zed).png
			...
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
		parser.add_argument('--offset', type=int, default=0, help='Offset of the scene id')
		print('Usage: python gendataset_from_files.py --config config_matterport3d.yaml --in_dir out_general --out_dir map_multisession_eval --scene_id 0 --start_indice 0')
		self.args = parser.parse_args()

		self.poses_odom = np.loadtxt(os.path.join(self.args.in_dir, 'poses_abs_odom.txt'))
		self.poses_gt = np.loadtxt(os.path.join(self.args.in_dir, 'poses_abs_gt.txt'))
		self.intrinsics = np.loadtxt(os.path.join(self.args.in_dir, 'intrinsics.txt'))
		self.start_indice = self.args.start_indice
			
	def setup_directories(self):
		base_path = os.path.join(self.args.out_dir, f's{self.args.scene_id:05d}')
		paths = [base_path]
		for i in range(self.args.num_split):
			paths.append(os.path.join(base_path, f'out_map{i+self.args.offset}'))
			paths.append(os.path.join(base_path, f'out_map{i+self.args.offset}/seq'))
		for path in paths:
			os.makedirs(path, exist_ok=True)
		self.base_path = base_path

	def run(self):
		total_len = len(self.poses_odom) 
		N = self.args.num_split
		path_segments = [
			(int(max(self.start_indice, i * total_len / N)), int(min(total_len, (i + 1) * total_len / N)))
			for i in range(N)
		]
		print(f'Total length of segment: {total_len}, split {N} segments')
		print('Segments: ', path_segments)

		for seg_id, (start_ind, end_ind) in enumerate(path_segments):
			print(f'Processing segment {seg_id} from {start_ind} to {end_ind}')
			seg_time_odom = np.empty((0, 2), dtype=object)
			seg_time_gt = np.empty((0, 2), dtype=object)
			seg_intrinsics = np.empty((0, 7), dtype=object)
			seg_poses_rel_odom = np.empty((0, 8), dtype=object)
			seg_poses_rel_gt = np.empty((0, 8), dtype=object)
			seg_poses_abs_gt = np.empty((0, 8), dtype=object)
			edges = np.empty((0, 3), dtype=object)

			for ind in range(start_ind, end_ind, self.args.step):
				cur_ind = int((ind - start_ind) / self.args.step)
				##### Time
				vec = np.empty((1, 2), dtype=object)
				vec[0, 0], vec[0, 1] = f'seq/{cur_ind:06d}.color.jpg', self.poses_odom[ind, 0]
				seg_time_odom = np.vstack((seg_time_odom, vec))

				vec = np.empty((1, 2), dtype=object)
				vec[0, 0], vec[0, 1] = f'seq/{cur_ind:06d}.color.jpg', self.poses_gt[ind, 0]
				seg_time_gt = np.vstack((seg_time_odom, vec))

				##### Intrinsics
				vec = np.empty((1, 7), dtype=object)
				vec[0, 0], vec[0, 1:5], vec[0, 5], vec[0, 6] = \
					f'seq/{cur_ind:06d}.color.jpg', self.intrinsics[ind, :4], \
					int(self.intrinsics[ind, 4]), int(self.intrinsics[ind, 5])
				seg_intrinsics = np.vstack((seg_intrinsics, vec))

				##### Poses in the absolute world frame of all frames (gt)
				vec = np.empty((1, 8), dtype=object)
				Twc = convert_vec_to_matrix(self.poses_gt[ind, 1:4], self.poses_gt[ind, 4:], 'xyzw')
				tsl, quat = convert_matrix_to_vec(np.linalg.inv(Twc), 'wxyz')
				vec[0, 0], vec[0, 1:5], vec[0, 5:] = f'seq/{cur_ind:06d}.color.jpg', quat, tsl
				seg_poses_abs_gt = np.vstack((seg_poses_abs_gt, vec))

				##### Poses in the relative world frame of segmented frames (odometry)
				vec = np.empty((1, 8), dtype=object)
				T_w2ct = convert_vec_to_matrix(self.poses_odom[ind, 1:4], self.poses_odom[ind, 4:], 'xyzw')
				# NOTE(gogojjh): the first reference frame is represented in the absolute world frame
				if seg_id == 0:
					T_wsegt2c = T_w2ct
				# NOTE(gogojjh): the other frames are represented in the relative world frame (define the origin of the first frame)
				else:
					Tw2c_segt = convert_vec_to_matrix(self.poses_odom[start_ind, 1:4], self.poses_odom[start_ind, 4:], 'xyzw')
					T_wsegt2c = np.linalg.inv(Tw2c_segt) @ T_w2ct
				tsl, quat = convert_matrix_to_vec(np.linalg.inv(T_wsegt2c), 'wxyz')
				vec[0, 0], vec[0, 1:5], vec[0, 5:] = f'seq/{cur_ind:06d}.color.jpg', quat, tsl
				seg_poses_rel_odom = np.vstack((seg_poses_rel_odom, vec))

				##### Poses in the relative world frame of segmented frames (gt)
				vec = np.empty((1, 8), dtype=object)
				T_w2ct = convert_vec_to_matrix(self.poses_gt[ind, 1:4], self.poses_gt[ind, 4:], 'xyzw')
				# NOTE(gogojjh): the first reference frame is represented in the absolute world frame
				if seg_id == 0:
					T_wsegt2c = T_w2ct
				# NOTE(gogojjh): the other frames are represented in the relative world frame (define the origin of the first frame)
				else:
					Tw2c_segt = convert_vec_to_matrix(self.poses_gt[start_ind, 1:4], self.poses_gt[start_ind, 4:], 'xyzw')
					T_wsegt2c = np.linalg.inv(Tw2c_segt) @ T_w2ct
				tsl, quat = convert_matrix_to_vec(np.linalg.inv(T_wsegt2c), 'wxyz')
				vec[0, 0], vec[0, 1:5], vec[0, 5:] = f'seq/{cur_ind:06d}.color.jpg', quat, tsl
				seg_poses_rel_gt = np.vstack((seg_poses_rel_gt, vec))

				##### Edges
				if cur_ind > 0:
					dis = np.linalg.norm(self.poses_odom[(cur_ind - 1) * self.args.step + start_ind, 1:4] - self.poses_odom[cur_ind * self.args.step + start_ind, 1:4])
					edges = np.vstack((edges, np.array([cur_ind - 1, cur_ind, dis])))

				##### Save images from segmented frames
				rgb_img_path = os.path.join(self.args.in_dir, 'seq', f'{ind:06d}.color.jpg')
				depth_img_path = os.path.join(self.args.in_dir, 'seq', f'{ind:06d}.depth.png')
				sem_img_path = os.path.join(self.args.in_dir, 'seq', f'{ind:06d}.semantic.png')
				new_rgb_img_path = os.path.join(self.base_path, f'out_map{seg_id+self.args.offset}/seq', \
					f'{cur_ind:06d}.color.jpg')
				new_depth_img_path = os.path.join(self.base_path, f'out_map{seg_id+self.args.offset}/seq', \
					f'{cur_ind:06d}.depth.png')			
				new_sem_img_path = os.path.join(self.base_path, f'out_map{seg_id+self.args.offset}/seq', \
					f'{cur_ind:06d}.semantic.png')
				os.system(f'cp {rgb_img_path} {new_rgb_img_path}')
				os.system(f'cp {depth_img_path} {new_depth_img_path}')
				try:
					os.system(f'cp {sem_img_path} {new_sem_img_path}')
				except:
					print(f'No semantic image at {sem_img_path}')

			np.savetxt(os.path.join(self.base_path, f'out_map{seg_id+self.args.offset}/timestamps.txt'), \
				seg_time_odom, fmt='%s %.9f')
			np.savetxt(os.path.join(self.base_path, f'out_map{seg_id+self.args.offset}/intrinsics.txt'), \
				seg_intrinsics, fmt='%s' + ' %.6f' * 4 + ' %d' * 2)
			np.savetxt(os.path.join(self.base_path, f'out_map{seg_id+self.args.offset}/poses.txt'), \
				seg_poses_rel_odom, fmt='%s' + ' %.6f' * 7)
			np.savetxt(os.path.join(self.base_path, f'out_map{seg_id+self.args.offset}/poses_rel_gt.txt'), \
				seg_poses_rel_gt, fmt='%s' + ' %.6f' * 7)
			np.savetxt(os.path.join(self.base_path, f'out_map{seg_id+self.args.offset}/poses_abs_gt.txt'), \
				seg_poses_abs_gt, fmt='%s' + ' %.6f' * 7)
			np.savetxt(os.path.join(self.base_path, f'out_map{seg_id+self.args.offset}/edge_list.txt'), \
				edges, fmt='%d %d %.6f')
		print('Finish generating dataset')
		# input()

if __name__ == '__main__':
	data_generator = DataGenerator()
	data_generator.setup_directories() 
	data_generator.run()

