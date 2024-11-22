# NOTE(gogojjh): Not tested

#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from utils_ros.tools_ros_msg_conversion import convert_pts_to_rospts, convert_rospts_to_pts
from utils_math.tools_eigen import convert_matrix_to_vec, convert_vec_to_matrix

import copy
import threading
import time

import open3d as o3d
import rospy
import ros_numpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import numpy as np
import tf
import tf.transformations
import argparse

# Global variables
global_map = None
initialized = False
T_map_to_odom = np.eye(4)
current_odom = None
current_scan = None
keyframe_scans = []

# Configuration constants
SCAN_DISTANCE_THRESHOLD = 3.0  # Distance threshold for keyframes
MAP_VOXEL_SIZE = 0.4
SCAN_VOXEL_SIZE = 0.1
FREQ_LOCALIZATION = 0.5  # Global localization frequency (HZ)
LOCALIZATION_TH = 0.95  # Threshold for global localization fitness
FOV = 6.28  # FOV(rad)
FOV_FAR = 30  # Farthest distance within FOV (meters)

def pose_to_matrix(pose_msg):
	"""Convert pose message to transformation matrix."""
	return np.matmul(
		tf.listener.xyz_to_mat44(pose_msg.pose.pose.position),
		tf.listener.xyzw_to_mat44(pose_msg.pose.pose.orientation),
	)

def perform_registration(pc_scan, pc_map, initial, scale):
	"""Perform ICP registration at a given scale."""
	print('Perform Registration')
	pc_map.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
	pc_scan.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
	result_icp = o3d.pipelines.registration.registration_icp(
		pc_scan, pc_map, 0.2, initial,
		o3d.pipelines.registration.TransformationEstimationPointToPlane(),
		o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
														  relative_rmse=1e-6,
														  max_iteration=200)
	)
	return result_icp.transformation, result_icp.fitness

def global_localization(pose_estimation):
	"""Perform global localization using scan-to-map matching."""
	global global_map, current_scan, current_odom, T_map_to_odom
	rospy.loginfo('Global localization by scan-to-map matching...')

	scan_to_be_mapped = copy.copy(current_scan)
	tic = time.time()

	# Registration
	toc = time.time()
	transformation, fitness = perform_registration(scan_to_be_mapped, global_map, initial=pose_estimation, scale=1)
	print('Time: {}'.format(toc - tic))

	if fitness > LOCALIZATION_TH:
		T_map_to_odom = transformation
		trans, quat = convert_matrix_to_vec(T_map_to_odom)
		header = Header()
		header.frame_id = 'map'
		header.stamp = current_odom.header.stamp
		odom = convert_vec_to_rosodom(trans, quat, header, 'base_link')
		pub_map_to_odom.publish(odom)
		return True, T_map_to_odom
	else:
		rospy.logwarn('Not match!!!!')
		rospy.logwarn('{}'.format(transformation))
		rospy.logwarn('fitness score:{}'.format(fitness))
		return False, T_map_to_odom

def voxel_down_sample(point_cloud, voxel_size):
	"""Downsample a point cloud using a voxel grid filter."""
	try:
		point_cloud_downsampled = point_cloud.voxel_down_sample(voxel_size)
	except:
		point_cloud_downsampled = o3d.geometry.voxel_down_sample(point_cloud, voxel_size)
	return point_cloud_downsampled

def initialize_global_map(args):
	"""Initialize the global map using the received point cloud message."""
	global global_map
	print('Loading global map...')
	if os.path.exists(args.path_global_map):
		global_map = o3d.io.read_point_cloud(args.path_global_map)
		global_map = voxel_down_sample(global_map, MAP_VOXEL_SIZE)
		print(global_map)
	else:
		print('Global map file not found!')
		exit(1)

def save_current_odom(odom_msg):
	"""Callback function to save the current odometry."""
	global current_odom
	current_odom = odom_msg

def save_current_scan(pc_msg):
	"""Callback function to save the current scan."""
	global current_scan, keyframe_scans
	pc = convert_rospts_to_pts(pc_msg)
	current_scan = o3d.geometry.PointCloud()
	current_scan.points = o3d.utility.Vector3dVector(pc[:, :3])

	# if len(keyframe_scans) == 0 or np.linalg.norm(pc[:, :3] - np.asarray(keyframe_scans[-1].points)) > SCAN_DISTANCE_THRESHOLD:
	#     keyframe_scans.append(current_scan)
	#     if len(keyframe_scans) > 10:
	#         keyframe_scans.pop(0)

def create_submap():
	"""Create a submap by merging keyframe scans."""
	submap = o3d.geometry.PointCloud()
	for scan in keyframe_scans:
		submap += scan
	submap = voxel_down_sample(submap, MAP_VOXEL_SIZE)
	return submap

def publish_pointcloud(pub_global_map, header, pc):
	pc_msg = convert_pts_to_rospts(header, pc)
	pub_global_map.publish(pc_msg)

def thread_localization():
	"""Thread function to perform periodic global localization."""
	global T_map_to_odom
	while True:
		if rospy.is_shutdown():
			break
		rospy.sleep(1 / FREQ_LOCALIZATION)
		submap = create_submap()
		global_localization(T_map_to_odom)

def thread_publish_global_map(global_map, pub_global_map):
	header = Header()
	header.frame_id = 'map'
	while True:
		if rospy.is_shutdown():
			break
		header.stamp = rospy.Time.now()
		publish_pointcloud(pub_global_map, header, np.asarray(global_map.points)[::3])
		rospy.sleep(0.5)

if __name__ == '__main__':
	"""Parser"""
	parser = argparse.ArgumentParser(description='Read a point cloud file and publish it as a global map.')
	parser.add_argument('--path_global_map', type=str, help='Path to the global map point cloud file.')
	args = parser.parse_args()

	"""ROS"""
	rospy.init_node('fast_lio_localization')
	rospy.loginfo('Localization Node Inited...')

	# Subscribers
	rospy.Subscriber('/cloud_registered_body', PointCloud2, save_current_scan, queue_size=1)
	rospy.Subscriber('/Odometry', Odometry, save_current_odom, queue_size=1)

	# Publishers
	pub_pc_in_map = rospy.Publisher('/cur_scan_in_map', PointCloud2, queue_size=1)
	pub_submap = rospy.Publisher('/submap', PointCloud2, queue_size=1)
	pub_map_to_odom = rospy.Publisher('/odometry_global', Odometry, queue_size=1)
	pub_global_map = rospy.Publisher('/map_global', PointCloud2, queue_size=1)

	"""Read global map"""
	initialize_global_map(args)
	t_publish = threading.Thread(target=thread_publish_global_map, args=(global_map, pub_global_map, ))
	t_publish.start()

	"""Initialize poses"""
	while not initialized:
		if rospy.is_shutdown():
			break
		rospy.logwarn('Waiting for initial pose...')
		
		pose_msg = rospy.wait_for_message('/initialpose', PoseWithCovarianceStamped)
		initial_pose = pose_to_matrix(pose_msg)
		trans, quat = convert_matrix_to_vec(initial_pose)
		if current_scan:
			initialized, T_map_scan = global_localization(initial_pose)
			trans, quat = convert_matrix_to_vec(T_map_scan)
			print('Initial poses: ', trans, quat)
		else:
			rospy.logwarn('First scan not received!')
	rospy.loginfo('Initialization successful!')

	t_loc = threading.Thread(target=thread_localization)
	t_loc.start()
	rospy.spin()

	t_loc.join()
	t_publish.join()
