import open3d as o3d
import numpy as np
import argparse
import cv2

def parse_arguments():
	"""
	Parses input arguments for the script.
	"""
	parser = argparse.ArgumentParser(description='Convert RGB and Depth images to Point Cloud')
	parser.add_argument('--input_rgb', type=str, required=True, help='Path to the RGB image')
	parser.add_argument('--input_depth', type=str, required=True, help='Path to the depth image')
	parser.add_argument('--output_pc', type=str, required=True, help='Path to save the output point cloud')
	parser.add_argument('--dataset_type', type=str, required=True, help='habitat, anymal_kinect, anymal_zed')
	parser.add_argument('--depth_scale', type=float, required=True, default='0.039', help='habitat: 0.039, anymal: 0.001')
	return parser.parse_args()

def create_camera_intrinsics(dataset_type):
	if dataset_type == 'habitat':
	  camera_matrix = np.array(
			[205.46963709898583, 0.0, 320.5, 
			0.0, 205.46963709898583, 180.5, 
			0.0, 0.0, 1.0]          
	  ).reshape(3, 3)
	if dataset_type == 'anymal_kinect':
	  camera_matrix = np.array(
			[607.9638061523438, 0.0, 638.83984375, 
			0.0, 607.9390869140625, 367.0916748046875, 
			0.0, 0.0, 1.0]
	  ).reshape(3, 3)
	if dataset_type == 'anymal_zed':
	  camera_matrix = np.array(
			[542.9769287109375, 0.0, 481.2997741699219, 
			0.0, 542.9769287109375, 271.8504638671875, 
			0.0, 0.0, 1.0]
	  ).reshape(3, 3)
	return camera_matrix

def load_images(rgb_path, depth_path):
	"""
	Loads the RGB and depth images from the specified paths.
	"""
	rgb_image = cv2.imread(rgb_path)
	depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
	return rgb_image, depth_image

def create_point_cloud(rgb_image, depth_image, camera_matrix, depth_scale):
	"""
	Creates a point cloud from the RGB and depth images using camera intrinsics.
	"""
	height, width = depth_image.shape
	fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2]

	i, j = np.indices((height, width))
	z = depth_image * depth_scale
	x = (j - cx) * z / fx
	y = (i - cy) * z / fy
	
	points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
	colors = (cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0).reshape(-1, 3)
	
	valid_mask = points[:, 2] > 0
	points = points[valid_mask]
	colors = colors[valid_mask]
	
	point_cloud = o3d.geometry.PointCloud()
	point_cloud.points = o3d.utility.Vector3dVector(points)
	point_cloud.colors = o3d.utility.Vector3dVector(colors)
	
	return point_cloud

def save_point_cloud(point_cloud, output_path):
	"""
	Saves the generated point cloud to the specified path.
	"""
	o3d.io.write_point_cloud(output_path, point_cloud)

def main():
	"""
	Main function to parse arguments, load images, create point cloud, and save it.
	"""
	args = parse_arguments()
	camera_matrix = create_camera_intrinsics(args.dataset_type)
	rgb_image, depth_image = load_images(args.input_rgb, args.input_depth)
	point_cloud = create_point_cloud(rgb_image, depth_image, camera_matrix, args.depth_scale)
	save_point_cloud(point_cloud, args.output_pc)

if __name__ == "__main__":
	main()
