'''
Author: David-Willo davidwillo@foxmail.com
Date: 2024-08-21 17:08:04
LastEditTime: 2024-08-21 18:01:46
LastEditors: David-Willo
Jinhao HE (David Willo), IADC HKUST(GZ)
Copyright (c) 2024 by davidwillo@foxmail.com, All Rights Reserved. 
'''

'''
[scenename]/                           # [input] Root directory named after a specific scene or dataset
│
├── xx.yaml                            # [input] Calibration file
│
├── [seqname]/                         # [input] Directory for a particular sequence within the scene
│   ├── pcd/                           # [output] Directory for point cloud data (.pcd files)
│   │   └── [stamp_ns].pcd             # Point cloud files named by timestamp in nanoseconds
│   │
│   ├── images_sampled/                # [output] Directory for processed or sampled images
│   │   └── [stamp_ns].jpg             # Sampled images, named by timestamp
│   │
│   ├── images_raw/                    # [output] Directory for raw images
│   │   └── [stamp_ns].jpg             # Raw images, named by timestamp
│   │
│   ├── lidar_ref.txt                  # [input] Text file, references for LiDAR data
│   ├── images_tum.txt                 # [output] Text file with image data in TUM format (potentially timestamps and filenames)
│   ├── images_sampled_tum.txt         # [output] Similar to images_tum.txt but for the sampled images
│   └── xxx.bag                        # [input] ROS bag file containing raw data for the sequence
'''

import os
import yaml
import numpy as np
import rosbag
from scipy.spatial.transform import Rotation as R
from sensor_msgs import point_cloud2
import open3d as o3d
import tf
from cv_bridge import CvBridge
import cv2
import math
import shutil
from tqdm import tqdm

def load_camera_parameters(yaml_file):
    with open(yaml_file, 'r') as f:
        camera_params = yaml.safe_load(f)

    fx = camera_params['projection_parameters']['fx']
    fy = camera_params['projection_parameters']['fy']
    cx = camera_params['projection_parameters']['cx']
    cy = camera_params['projection_parameters']['cy']
    k1 = camera_params['distortion_parameters']['k1']
    k2 = camera_params['distortion_parameters']['k2']
    p1 = camera_params['distortion_parameters']['p1']
    p2 = camera_params['distortion_parameters']['p2']

    width = camera_params['image_width']
    height = camera_params['image_height']

    intrinsics = [fx, fy, cx, cy, width, height]
    distortion = [k1, k2, p1, p2]

    R_cl = np.array(camera_params['extrinsicRotation']['data']).reshape(3, 3)
    t_cl = np.array(camera_params['extrinsicTranslation']['data']).reshape(3)

    R_lc = np.linalg.inv(R_cl)
    t_lc = -R_lc @ t_cl
    return intrinsics, distortion, R_lc, t_lc

def find_files(directory, extension):
    """ Find the first file with the given extension. """
    for file in os.listdir(directory):
        if file.endswith(extension):
            return os.path.join(directory, file)
    return None
    
def load_tum_trajectory(filename):
    """ Load TUM trajectory file """
    trajectory = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            timestamp = float(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            trajectory[timestamp] = (np.array([tx, ty, tz]), np.array([qx, qy, qz, qw]))
    return trajectory

def interpolate_pose(trajectory, timestamp):
    """ Interpolate pose at a given timestamp from a trajectory loaded from TUM format.
    
    Args:
        trajectory (dict): A dictionary where the key is the timestamp and the value is a tuple of position and orientation.
        timestamp (float): The specific timestamp to interpolate the pose for.
    
    Returns:
        list: Interpolated pose as [tx, ty, tz, qx, qy, qz, qw].
    """
    
    # Extract keys and sort them
    timestamps = sorted(trajectory.keys())
    # Handle boundary cases
    if timestamp <= timestamps[0]:
        return np.hstack(trajectory[timestamps[0]]).tolist()
    if timestamp >= timestamps[-1]:
        return np.hstack(trajectory[timestamps[-1]]).tolist()
    
    # Find the appropriate interval for interpolation
    for i in range(len(timestamps) - 1):
        t0, t1 = timestamps[i], timestamps[i + 1]
        if t0 <= timestamp <= t1:
            pose0, pose1 = trajectory[t0], trajectory[t1]
            alpha = (timestamp - t0) / (t1 - t0)
            trans0, trans1 = pose0[0], pose1[0]
            quat0, quat1 = pose0[1], pose1[1]
            
            # Interpolate translation
            trans = (1 - alpha) * trans0 + alpha * trans1
            
            # Interpolate quaternion using SLERP
            quat = tf.transformations.quaternion_slerp(quat0, quat1, alpha)
            
            # Return as a flat list
            return np.hstack((trans, quat)).tolist()
    
    # In case no interval was found (should not happen)
    return None

def apply_lidar_camera_extrinsics(interpolate_pose, R_lc, t_lc):  
    camera_poses = []
    for ipose in interpolate_pose:
        if ipose is not None:
            tx, ty, tz, qx, qy, qz, qw = ipose
            lidar_rotation = R.from_quat([qx, qy, qz, qw])
            lidar_translation_vector = np.array([tx, ty, tz])

            camera_rotation = lidar_rotation * R.from_matrix(R_lc)
            camera_translation_vector = lidar_rotation.apply(t_lc) + lidar_translation_vector

            camera_pose = camera_translation_vector.tolist() + camera_rotation.as_quat().tolist()
            camera_poses.append(camera_pose)
    return camera_poses


def transform_pointcloud(cloud, pose):
    """ Transform point cloud using the given pose """
    translation, quaternion = pose
    rot = R.from_quat(quaternion)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rot.as_matrix()
    transformation_matrix[:3, 3] = translation
    
    # Apply transformation
    cloud.transform(transformation_matrix)
    return cloud

def transform_and_extract_pcds_from_bag(bagfile, trajectory, cache_dir):
    """ Read rosbag, transform point clouds and save them """
    bag = rosbag.Bag(bagfile)
    # for topic, msg, t in tqdm(bag.read_messages(topics=['/livox/lidar_192_168_8_182']), 'extracting pcds'):  # Change '/lidar_topic' to your LIDAR data topic
    for topic, msg, t in bag.read_messages(topics=['/top/rslidar_points']):  # Change '/lidar_topic' to your LIDAR data topic
        stamp = msg.header.stamp.to_sec()
        stamp_ns = msg.header.stamp.to_nsec()
        closest_time = min(trajectory.keys(), key=lambda x: abs(x - stamp))
        pose = trajectory[closest_time]
        
        # Convert ROS PointCloud2 to Open3D point cloud
        pc = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(pc))
        o3d_cloud = o3d.geometry.PointCloud()
        try:
            o3d_cloud.points = o3d.utility.Vector3dVector(points)
            
            # Transform the point cloud
            transformed_cloud = transform_pointcloud(o3d_cloud, pose)
            
            # Save to file
            #filename = os.path.join(cache_dir, f"{int(closest_time*1e9)}.pcd")
            filename = os.path.join(cache_dir, f"{int(stamp_ns)}.pcd")
            o3d.io.write_point_cloud(filename, transformed_cloud)
        except Exception as e:
            # Handles any exception
            print(f"An error occurred: {e}")
    
    bag.close()

def extract_images_and_save(bag_file, output_dir, topic, intrinsics, distortion, skip=False):
    os.makedirs(output_dir, exist_ok=True)
    bag = rosbag.Bag(bag_file, 'r')
    bridge = CvBridge()
    frame_paths = []
    timestamps = []
    timestamps_ns = []
    fx, fy, cx, cy, width, height = intrinsics
    k1, k2, p1, p2 = distortion
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    D = np.array([k1, k2, 0, p1, p2])
    
    # Precompute the undistortion and rectification maps
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (width, height), cv2.CV_32FC1)
    for i, (topic, msg, t) in tqdm(enumerate(bag.read_messages(topics=[topic])), 'extract images'):
        frame_path = os.path.join(output_dir, f'{t.to_nsec()}.jpg')
        frame_paths.append(frame_path)
        timestamps.append(t.to_sec())
        timestamps_ns.append(t.to_nsec())
        if not skip:
            cv_image = bridge.compressed_imgmsg_to_cv2(msg)
            resized_image = cv2.resize(cv_image, (width, height))
            #undistorted_image = cv2.undistort(resized_image, K, D, None, K)
            undistorted_image = cv2.remap(resized_image, map1, map2, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(frame_path, undistorted_image)
    bag.close()
    return frame_paths, timestamps, timestamps_ns

def compute_pose_difference(pose1, pose2):
    r1, t1 = R.from_quat(pose1[3:]), np.array(pose1[:3])
    r2, t2 = R.from_quat(pose2[3:]), np.array(pose2[:3])
    rotation_diff = R.inv(r1) * r2
    translation_diff = np.linalg.norm(t1 - t2)
    return  translation_diff, rotation_diff.magnitude()*180/math.pi

def sample_and_copy_images(frame_paths, img_timestamps, camera_poses, images_sample_dir, translation_thresh_m, rotation_thresh_deg):
    if not os.path.exists(images_sample_dir):
        os.makedirs(images_sample_dir)
        
    camera_poses_sampled = []
    img_timestamps_sampled = []
    previous_pose = camera_poses[0]
    previous_frame_path = frame_paths[0]
    previous_stamp = img_timestamps[0]

    # Always include the first image
    shutil.copy(previous_frame_path, os.path.join(images_sample_dir, os.path.basename(previous_frame_path)))
    camera_poses_sampled.append(previous_pose)
    img_timestamps_sampled.append(previous_stamp)

    for current_frame_path, current_pose, current_stamp in tqdm(zip(frame_paths[1:], camera_poses[1:], img_timestamps[1:]), 'sample images'):
        trans_diff, rot_diff = compute_pose_difference(previous_pose, current_pose)
        
        if trans_diff >= translation_thresh_m or rot_diff >= rotation_thresh_deg:
            shutil.copy(current_frame_path, os.path.join(images_sample_dir, os.path.basename(current_frame_path)))
            camera_poses_sampled.append(current_pose)
            img_timestamps_sampled.append(current_stamp)
            previous_pose = current_pose  # Update the reference pose to the current pose

    return img_timestamps_sampled, camera_poses_sampled

def export_tum_pose(timestamps, poses, tum_file_path):
    with open(tum_file_path, 'w') as tum_file:
        # Assuming camera_poses are computed; this is a placeholder
        for ts, pose in zip(timestamps, poses):
            tum_file.write(f"{ts} {pose[0]} {pose[1]} {pose[2]} {pose[3]} {pose[4]} {pose[5]} {pose[6]}\n")

def process_sequence(scene_dir, seqname):
    seq_path = os.path.join(scene_dir, seqname)
    bag_path = find_files(seq_path, '.bag')
    yaml_path = find_files(scene_dir, '.yaml')
    lidar_ref_path = os.path.join(seq_path, 'lidar_ref.txt')
    img_tum_file_path = os.path.join(seq_path, 'images_tum.txt')
    img_sampled_tum_file_path = os.path.join(seq_path, 'images_sampled_tum.txt')
    image_topic = "/usb_cam/image_raw/compressed"
    skip_images = False
    translation_thresh_m = 1
    rotation_thresh_deg = 5

    print(f"Processing sequence in: {seq_path}")
    
    if not bag_path or not yaml_path:
        print(f"Missing files for sequence {seqname} in {scene_dir}")
        return
    
    
    # Create directories if they don't exist
    pcd_dir = os.path.join(seq_path, 'pcd')
    images_raw_dir = os.path.join(seq_path, 'images_raw')
    images_sample_dir = os.path.join(seq_path, 'images_sampled')
    os.makedirs(pcd_dir, exist_ok=True)
    os.makedirs(images_sample_dir, exist_ok=True)
    os.makedirs(images_raw_dir, exist_ok=True)

    # Here you would process the bag file to extract and transform point clouds, save images, etc.
    # This would require actual functions to read ROS bag files, extract data, and transform it.
    # The following is a placeholder for the actual data processing logic.
    # Assuming camera extrinsics extraction and other steps are similar to your initial example...
    
    intrinsics, distortion, R_lc, t_lc = load_camera_parameters(yaml_path)
    trajectory = load_tum_trajectory(lidar_ref_path)
    transform_and_extract_pcds_from_bag(bag_path, trajectory, pcd_dir)
    frame_paths, img_timestamps, img_timestamps_ns = extract_images_and_save(bag_path, images_raw_dir, image_topic, intrinsics, distortion, skip=skip_images)
    interpolate_poses = [interpolate_pose(trajectory, ts) for ts in img_timestamps]
    camera_poses = apply_lidar_camera_extrinsics(interpolate_poses, R_lc, t_lc)
    img_timestamps_sampled, camera_poses_sampled = sample_and_copy_images(frame_paths, img_timestamps, camera_poses, images_sample_dir, translation_thresh_m, rotation_thresh_deg)
    
    # Example: Save an image pose file in TUM format  
    export_tum_pose(img_timestamps, camera_poses, img_tum_file_path) 
    export_tum_pose(img_timestamps_sampled, camera_poses_sampled, img_sampled_tum_file_path) 
    

    



def main():
    base_dir = '/media/host/OneTouch/APMP_dataset'
    # scenes_to_process = ['cainiao', 'huoniao', 'fantang', 'tiyuguan']
    scenes_to_process = ['lab']
    for scene in scenes_to_process:
        scene_dir = os.path.join(base_dir, scene)
        seqnames = [d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))]
        for seqname in seqnames:
            process_sequence(scene_dir, seqname)

if __name__ == "__main__":
    main()
