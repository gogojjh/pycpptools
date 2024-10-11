"""
Author: Jianhao Jiao
Date: 2024-07-10
Description: This script generates camera data including RGB images, depth images, semantic images, and poses a ROS bag
Version: 1.0
Usage: python gencamera_from_bag.py --config /path/config_anymal.yaml --out_dir /data_anymal
"""

"""Format of generate dataset
ucl_east/
    seq/
        000000.color.png
        000000.depth.png (mm)
		000000.semantic.png
    poses.txt (TUM format: timestamp tx ty tz qx qy qz qw)
    intrinsics.txt (format: fx fy cx cy width height)
"""

"""Format of 3dmatch dataset
ucl_east/
    seq-01/
        frame-000000.color.png
        frame-000000.depth.png (mm)
		frame-000000.pose.txt (format: 4x4 transformation matrix)
    camera-intrinsics.txt (format: 3x3 intrinsics matrix)
"""

"""Format of euroc dataset
ucl_east/
    seq-01/
        cam0/data/
			timestamp1.png (nsec)
			timestamp2.png (nsec)
			...
		cam1/data/
			timestamp1.png (nsec)
			timestamp2.png (nsec)
			...
"""

import os
import argparse
import numpy as np

import rospy
import message_filters
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped
from tf import TransformListener
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image, CompressedImage, CameraInfo

from pycpptools.src.python.utils_ros.tools_ros_msg_conversion import convert_rosodom_to_vec
from pycpptools.src.python.utils_math.tools_eigen import add_gaussian_noise_to_pose, compute_relative_dis, convert_vec_to_matrix, convert_matrix_to_vec
from scipy.spatial.transform import Rotation
import yaml

bridge = CvBridge()

class DataGenerator:
	def __init__(self):
		# Initialize argument parser and add arguments
		parser = argparse.ArgumentParser(description="Camera data collector for synchronized RGBD images and poses.")
		parser.add_argument('--config', type=str, default='/dataset/config_anymal.yaml', help='Path to configuration file')
		parser.add_argument('--out_dir', type=str, default='/tmp', help='Path to save data')
		self.args = parser.parse_args()

		with open(self.args.config, 'r') as file:
			config = yaml.safe_load(file)

		self.out_data_format = config['out_data_format']

		# Set image type and conversion function based on dataset type
		self.RGB_IMAGE_TYPE = CompressedImage if 'real' in config['dataset_type'] else Image
		self.RGB_CV_FUNCTION = bridge.compressed_imgmsg_to_cv2 if 'real' in config['dataset_type'] else CvBridge().imgmsg_to_cv2

		# Initialize ROS node
		rospy.init_node('data_generator')

		# Setup subscribers and synchronizer for image and odometry topics
		if self.out_data_format == 'euroc':
			# Subscribe to stereo camera topics
			left_rgb_sub = message_filters.Subscriber(config['left_rgb_topic'], self.RGB_IMAGE_TYPE)
			right_rgb_sub = message_filters.Subscriber(config['right_rgb_topic'], self.RGB_IMAGE_TYPE)
			ts = message_filters.ApproximateTimeSynchronizer([left_rgb_sub, right_rgb_sub], 100, 0.1, allow_headerless=True)
			ts.registerCallback(self.stereo_image_callback)
		else:
			camera_info_sub = message_filters.Subscriber(config['camera_info_topic'], CameraInfo)
			rgb_sub = message_filters.Subscriber(config['rgb_topic'], self.RGB_IMAGE_TYPE)
			depth_sub = message_filters.Subscriber(config['depth_topic'], Image)
			semantic_sub = message_filters.Subscriber(config['semantic_topic'], self.RGB_IMAGE_TYPE)
			base_odom_sub = message_filters.Subscriber(config['odometry_topic'], Odometry)
			ts = message_filters.ApproximateTimeSynchronizer([camera_info_sub, rgb_sub, depth_sub, semantic_sub, base_odom_sub], 100, 0.1, allow_headerless=True)
			ts.registerCallback(self.image_callback)

		# NOTE(gogojjh): transform the SLAM poses on the base_frame to the camera_frame
		quat_base_cam = np.array(config['quat_base_cam'])
		trans_base_cam = np.array(config['trans_base_cam'])
		self.T_base_cam = convert_vec_to_matrix(trans_base_cam, quat_base_cam, 'xyzw')

		# Setup directories for saving data
		self.setup_directories()

		# Initialize camera pose
		self.camera_poses = np.empty((0, 8))
		self.camera_intri = np.empty((0, 6))

		# Keep the node running
		rospy.spin()

	def image_callback(self, camera_info, rgb_image, depth_image, semantic_image, base_odom):
		print(f'image_callback: {self.camera_poses.shape[0]}')
		timestamp = rgb_image.header.stamp

		# Extract camera info
		vec = np.array([camera_info.K[0], camera_info.K[4], camera_info.K[2], camera_info.K[5], camera_info.width, camera_info.height])
		self.camera_intri = np.vstack([self.camera_intri, vec])

		# Convert and save RGB, depth, and semantic image
		cv_rgb_image = self.RGB_CV_FUNCTION(rgb_image, "bgr8")
		cv_depth_image = bridge.imgmsg_to_cv2(depth_image, "passthrough")
		if depth_image.encoding == "32FC1":
			cv_depth_image = np.nan_to_num(cv_depth_image, nan=0.0, posinf=1e3, neginf=0.0)
			cv_depth_image = (cv_depth_image * 1000).astype(np.uint16)
		if depth_image.encoding == "mono8":
			cv_depth_image = (cv_depth_image / 25.641025 * 1000).astype(np.uint16) 
		cv_semantic_image = self.RGB_CV_FUNCTION(semantic_image, "bgr8")

		if self.out_data_format == '3dmatch':
			cv2.imwrite(f'{self.base_path}/seq-01/{self.suffix}{self.camera_poses.shape[0]:06d}.color.png', cv_rgb_image)
			cv2.imwrite(f'{self.base_path}/seq-01/{self.suffix}{self.camera_poses.shape[0]:06d}.depth.png', cv_depth_image)
		else:
			cv2.imwrite(f'{self.base_path}/seq/{self.suffix}{self.camera_poses.shape[0]:06d}.color.jpg', cv_rgb_image)
			cv2.imwrite(f'{self.base_path}/seq/{self.suffix}{self.camera_poses.shape[0]:06d}.depth.png', cv_depth_image)
			cv2.imwrite(f'{self.base_path}/seq/{self.suffix}{self.camera_poses.shape[0]:06d}.semantic.png', cv_semantic_image)

		# Convert odometry to translation and quaternion
		trans, quat = convert_rosodom_to_vec(base_odom, 'xyzw')
		T_w_base = convert_vec_to_matrix(trans, quat, 'xyzw')
		T_w_cam = T_w_base @ self.T_base_cam
		trans, quat = convert_matrix_to_vec(T_w_cam, 'xyzw')

		# output format: timestamp, tx, ty, tz, qx, qy, qz, qw
		vec = np.hstack((np.array(timestamp.to_sec()), trans, quat))
		self.camera_poses = np.vstack([self.camera_poses, vec])

		if self.out_data_format == '3dmatch':
			path = f'{self.base_path}/seq-01/{self.suffix}{self.camera_poses.shape[0]:06d}.pose.txt'
			np.savetxt(path, T_w_cam, fmt='%.5f')

	def stereo_image_callback(self, left_rgb_image, right_rgb_image):
		print(f'stereo_image_callback: {self.camera_poses.shape[0]}')
		left_timestamp = left_rgb_image.header.stamp
		right_timestamp = right_rgb_image.header.stamp
		cv_image = self.RGB_CV_FUNCTION(left_rgb_image, "bgr8")
		cv2.imwrite(f'{self.base_path}/cam0/data/{self.suffix}{int(left_timestamp.secs*1e9 + left_timestamp.nsecs):19d}.png', cv_image)
		cv_image = self.RGB_CV_FUNCTION(right_rgb_image, "bgr8")
		cv2.imwrite(f'{self.base_path}/cam1/data/{self.suffix}{int(right_timestamp.secs*1e9 + right_timestamp.nsecs):19d}.png', cv_image)

	def setup_directories(self):
		base_path = os.path.join(self.args.out_dir, f'out_{self.out_data_format}')
		if self.out_data_format == "general":
			paths = [self.args.out_dir, base_path, f'{base_path}/seq']
			for path in paths:
				os.makedirs(path, exist_ok=True)
			self.base_path = base_path
			self.suffix = ''
		elif self.out_data_format == "euroc":
			paths = [self.args.out_dir, base_path, f'{base_path}/cam0/data', f'{base_path}/cam1/data']
			for path in paths:
				os.makedirs(path, exist_ok=True)
			self.base_path = base_path
			self.suffix = ''
		else:
			paths = [self.args.out_dir, base_path, f'{base_path}/seq-01']
			for path in paths:
				os.makedirs(path, exist_ok=True)
			self.base_path = base_path
			self.suffix = 'frame-'

if __name__ == '__main__':
	data_generator = DataGenerator()
	if rospy.is_shutdown():
		print('ROS Shutdown, save data to {}'.format(data_generator.base_path))
		if data_generator.out_data_format == 'general':
			np.savetxt(os.path.join(data_generator.base_path, 'poses.txt'), data_generator.camera_poses, fmt='%.5f')
			np.savetxt(os.path.join(data_generator.base_path, 'intrinsics.txt'), data_generator.camera_intri, fmt='%.5f')
		elif data_generator.out_data_format == '3dmatch':
			vec = data_generator.camera_intri[0, :4]
			K = np.array([[vec[0], 0, vec[2]], [0, vec[1], vec[3]], [0, 0, 1]])
			np.savetxt(os.path.join(data_generator.base_path, 'camera-intrinsics.txt'), K, fmt='%.5f')
			