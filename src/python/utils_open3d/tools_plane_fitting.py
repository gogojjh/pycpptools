#! /usr/bin/env python

import open3d as o3d
import argparse
import numpy as np
import math

def fit_plane_to_pcd(path_input_pcd):
	# Load the point cloud from file
	pcd = o3d.io.read_point_cloud(path_input_pcd)
	
	# Fit a plane to the point cloud using RANSAC
	plane_model, inliers = pcd.segment_plane(distance_threshold=0.03,
																					 ransac_n=3,
																					 num_iterations=1000)
	
	# Extract the normal vector from the plane model
	[a, b, c, d] = plane_model
	normal_vector = np.array([a, b, c])

	# Extract inlier points
	inlier_cloud = pcd.select_by_index(inliers)
	inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Paint inliers red
	
	# Extract outlier points
	outlier_cloud = pcd.select_by_index(inliers, invert=True)
	outlier_cloud.paint_uniform_color([0, 0.0, 1.0])
	
	return normal_vector, inlier_cloud, outlier_cloud

def main():
	parser = argparse.ArgumentParser(description="Fit a plane to a point cloud using RANSAC and return the normal vector.")
	parser.add_argument('--path_input_pcd', type=str, help="Path to the input point cloud file (.pcd, .ply, etc.)")
	args = parser.parse_args()
	
	normal_vector, inlier_cloud, outlier_cloud = fit_plane_to_pcd(args.path_input_pcd)
	print(f"Normal vector of the fitted plane: {normal_vector}")

	roll  = math.acos(np.dot(normal_vector, np.array([1.0, 0.0, 0.0]))) / math.pi * 180.0
	pitch = math.acos(np.dot(normal_vector, np.array([0.0, 1.0, 0.0]))) / math.pi * 180.0
	yaw   = math.acos(np.dot(normal_vector, np.array([0.0, 0.0, 1.0]))) / math.pi * 180.0
	print(f"roll: {roll:03f}, pitch: {pitch:03f}, yaw: {yaw:03f}")
	# roll: 85.012187, pitch: 87.674082, yaw: 5.505962

	# Visualize the point cloud and the fitted plane
	o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

if __name__ == "__main__":
		main()
