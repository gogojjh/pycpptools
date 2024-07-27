#! /usr/bin/env python

import os
import argparse
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation
import copy

def estimate_pose_icp(source, target, current_transformation):
  threshold = 0.2
  # print("Point-to-Plane ICP registration")

  # print("Downsample with a voxel size %.2f" % radius)
  # radius = 0.005
  # source_down = source.voxel_down_sample(radius)
  # target_down = target.voxel_down_sample(radius)

  # print("Estimate normal.")
  source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
  target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

  # print("Applying point-to-plane registration")
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

# target is fixed, source needs to be transformted to align with target
def draw_registration_result(source, target, transformation):
    radius = 0.01 
    source_down = source.voxel_down_sample(radius)
    target_down = target.voxel_down_sample(radius)
    source_down.paint_uniform_color([1, 0.706, 0])
    target_down.paint_uniform_color([0, 0.651, 0.929])
    source_down.transform(transformation)
    o3d.visualization.draw_geometries([source_down, target_down], 
                                      zoom=1.0,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])

def crop_point_cloud(pcd):
  xyz = np.asarray(pcd.points)
  indices = np.abs(xyz[:, 0]) <= 50.0
  xyz = xyz[indices]
  indices = np.abs(xyz[:, 1]) <= 40.0
  xyz = xyz[indices]  
  indices = np.abs(xyz[:, 2]) <= 4.0
  xyz = xyz[indices]    
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(xyz)
  # o3d.visualization.draw_geometries([pcd])
  return pcd    

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Estimate camera pose using Open3D.')
  parser.add_argument('--target_pcd', help='target_pcd as the reference frame', default='target.pcd') # reference frame
  parser.add_argument('--src_pcd', help='src_pcd', default='src.pcd')
  args = parser.parse_args()

  # Read point clouds
  pcd_target = o3d.io.read_point_cloud(args.target_pcd)
  pcd_src = o3d.io.read_point_cloud(args.src_pcd)

  # Set initial guess 
  ##### To be changed
  T_ini = np.array([0.757572, -0.606689, 0.240860, -6.709962,
                    0.585685, 0.794683, 0.159539, 5.117188,
                    -0.288198, 0.020206, 0.957358, 0.707165,
                    0.000000, 0.000000, 0.000000, 1.000000]).reshape(4, 4)
  print('Initial Transform:\n', T_ini)
  
  # Crop point cloud
  # pcd_target = crop_point_cloud(pcd_target)  
  pcd_src.transform(T_ini)
  # pcd_src = crop_point_cloud(pcd_src)
  pcd_src.transform(np.linalg.inv(T_ini))

  # Downsample
  radius = 0.4
  pcd_src = pcd_src.voxel_down_sample(radius)  
  pcd_target = pcd_target.voxel_down_sample(radius)

  # Registration
  draw_registration_result(pcd_src, pcd_target, T_ini)
  T = estimate_pose_icp(pcd_src, pcd_target, T_ini)
  draw_registration_result(pcd_src, pcd_target, T)

  # Result
  T_result = copy.copy(T)
  quat_result = Rotation.from_matrix(T_result[:3, :3]).as_quat()
  trans_result = T_result[:3, 3]
  print('quat_opt: ', quat_result, '\ntrans_opt: ', trans_result.T)