import os
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
import copy
from PIL import Image

def estimate_pose_icp(source, target, current_transformation):
  threshold = 1.0

  source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
  target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

  result_icp = o3d.pipelines.registration.registration_icp(
    source, target, threshold, current_transformation, 
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                            relative_rmse=1e-6,
                            max_iteration=200))
  current_transformation = result_icp.transformation
  print('inlier_rmse: {}m'.format(result_icp.inlier_rmse))
  print('current_transformation:\n{}'.format(current_transformation))
  return result_icp.transformation

def draw_registration_result(source, target, transformation):
  radius = 0.1
  source_down = source.voxel_down_sample(radius)
  target_down = target.voxel_down_sample(radius)
  source_down.paint_uniform_color([0.8, 0, 0])
  target_down.paint_uniform_color([0, 0, 0])
  source_down.transform(transformation)
  o3d.visualization.draw_geometries([source_down, target_down], 
                    zoom=1.0,
                    front=[-0.2458, -0.8088, 0.5342],
                    lookat=[1.7745, 2.2305, 0.9787],
                    up=[0.3109, -0.5878, -0.7468])

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

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Estimate camera pose using Open3D.')
  parser.add_argument('--tar_depth', help='path to the first depth image as the reference frame')
  parser.add_argument('--src_depth', help='path to the second depth image')
  args = parser.parse_args()

  # Camera intrinsic
  K = np.array([205.469640000, 0.000000000, 320.00000000, 0.000000000, 205.469640000, 180.00000000, 0.000000000, 0.000000000, 1.000000000]).reshape(3, 3)
  img_size = (640, 360)
  
  # Read RGBD images
  depth_img1 = np.array(Image.open(args.tar_depth)) / 1000.0
  depth_points = depth_image_to_point_cloud(depth_img1, K, img_size)
  pcd_tar = o3d.geometry.PointCloud()
  pcd_tar.points = o3d.utility.Vector3dVector(depth_points)
  o3d.io.write_point_cloud('/Rocket_ssd/dataset/tmp/pcd_tar.pcd', pcd_tar)

  depth_img2 = np.array(Image.open(args.src_depth)) / 1000.0
  depth_points = depth_image_to_point_cloud(depth_img2, K, img_size)
  pcd_src = o3d.geometry.PointCloud()
  pcd_src.points = o3d.utility.Vector3dVector(depth_points)
  o3d.io.write_point_cloud('/Rocket_ssd/dataset/tmp/pcd_src.pcd', pcd_src)

  # Set initial transformation guess
  T_ini = np.eye(4)
  print('Initial Transform:\n', T_ini)

  # Downsample point clouds
  radius = 0.2
  pcd_src = pcd_src.voxel_down_sample(radius)
  pcd_tar = pcd_tar.voxel_down_sample(radius)

  # Registration
  # draw_registration_result(pcd_src, pcd_tar, T_ini)
  T = estimate_pose_icp(pcd_src, pcd_tar, T_ini)
  draw_registration_result(pcd_src, pcd_tar, T)

  # Result
  T_result = copy.copy(T)
  quat_result = Rotation.from_matrix(T_result[:3, :3]).as_quat()
  trans_result = T_result[:3, 3]
  print('quat_opt: ', quat_result, '\ntrans_opt: ', trans_result.T)
