'''
Author: David-Willo davidwillo@foxmail.com
Date: 2024-08-21 17:08:04
LastEditTime: 2024-08-21 18:01:46
LastEditors: David-Willo
Jinhao HE (David Willo), IADC HKUST(GZ)
Copyright (c) 2024 by davidwillo@foxmail.com, All Rights Reserved. 
'''

'''
script to reformat region data to map-free-loc style, pick ref image for testing and dedupe
[region_dir]/                          # [input] Root directory for region selection directory
│
├── [seqname]/                         # [input] Directory for a particular sequence within the scene
│   │
│   ├── images/                        # [input] Directory for processed or sampled images
│   │   └── [datasetscene_seq_stamp_ns].jpg                          # images
│   │   └── [datasetscene_seq_stamp_ns].ray_neighbor.png             # depths
│   │
│   └──  region_data.json               # [input] json file, contains image poses and intrinsics
'''

import os
import json
import math
import numpy as np
import shutil
from scipy.spatial.transform import Rotation as R
from PIL import Image
from scipy.spatial import cKDTree
from scipy.ndimage import grey_dilation
import cupy as cp
from concurrent.futures import ThreadPoolExecutor


def depth_image_to_point_cloud(depth_img, intrinsics):
    """Convert a depth image to a point cloud."""
    fx, fy, cx, cy, width, height = intrinsics
    i, j = np.indices((height, width))
    z = depth_img
    x = (j - cx) * z / fx
    y = (i - cy) * z / fy
    return np.stack((x, y, z), axis=-1).reshape(-1, 3)

def transform_point_cloud(points, pose):
    """Transform point cloud by pose."""
    rotation = R.from_quat(pose[3:])
    transformed_points = rotation.apply(points.reshape(-1, 3)) + pose[:3]
    return transformed_points.reshape(points.shape)

def compute_relative_pose(pose1, pose2):
    # Decompose poses
    r1, t1 = R.from_quat(pose1[3:]), np.array(pose1[:3])
    r2, t2 = R.from_quat(pose2[3:]), np.array(pose2[:3])
    
    # Compute relative transformation
    relative_rotation = R.inv(r1) * r2
    relative_translation = R.inv(r1).apply(t2 - t1)
    return np.concatenate((relative_translation, relative_rotation.as_quat()))

def project_point_cloud(points, intrinsics):
    """Project a point cloud onto an image plane."""
    fx, fy, cx, cy, width, height = intrinsics
    z = points[:, 2]
    x = (points[:, 0] * fx / z + cx).astype(np.int32)
    y = (points[:, 1] * fy / z + cy).astype(np.int32)

    depth_image = np.zeros((height, width))
    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (z > 0)
    depth_image[y[valid_mask], x[valid_mask]] = z[valid_mask]
    return depth_image

def compute_hit_mask(proj_depth_map, depth_img1, valid_mask):
    height, width = proj_depth_map.shape
    hit_mask = np.zeros_like(proj_depth_map, dtype=bool)

    # Create padded versions of depth_img1 and valid_mask to handle boundary conditions
    padded_depth = np.pad(depth_img1, pad_width=1, mode='constant', constant_values=0)
    padded_valid = np.pad(valid_mask, pad_width=1, mode='constant', constant_values=False)

    # Iterate over each possible neighborhood shift
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            # Extract shifted neighborhoods
            neighborhood = padded_depth[1 + di:height + 1 + di, 1 + dj:width + 1 + dj]
            valid_neighborhood = padded_valid[1 + di:height + 1 + di, 1 + dj:width + 1 + dj]
            
            # Compute mask where neighborhood depth is close to projected depth and is valid
            close_depth = np.abs(neighborhood - proj_depth_map) < 0.2
            valid_depth = neighborhood > 0
            condition = close_depth & valid_depth & valid_neighborhood

            # Update hit mask where any of the conditions are met
            hit_mask = hit_mask | condition

    return hit_mask


def compute_overlap(depth_img1, proj_depth_map):
    import matplotlib.pyplot as plt
    """Compute the overlap between two depth images."""
    height, width = depth_img1.shape
    valid_origin = depth_img1 > 0
    valid_mask = proj_depth_map > 0
    valid_pixel = np.abs(proj_depth_map[valid_mask] - depth_img1[valid_mask]) < 0.1
    # Initialize the hit mask
    # hit_mask = np.zeros_like(depth_img1, dtype=bool)
    # Compute the hit mask using the optimized function
    hit_mask = compute_hit_mask(proj_depth_map, depth_img1, valid_mask)

    # Calculate the number of valid pixels
    num_hits = np.sum(hit_mask)
    
    #print(f"valid origin { np.sum(valid_origin)} valid reproject { np.sum(valid_mask)} hit {np.sum(num_hits)}")
    # Plotting
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure and a set of subplots
    # # Plot the first depth image
    # ax[0].imshow(depth_img1, cmap='gray')
    # ax[0].set_title('Depth Image 1')
    # ax[0].axis('off')  # Hide axes ticks

    # # Plot the second depth image
    # ax[1].imshow(proj_depth_map  , cmap='gray')
    # ax[1].set_title('Depth Image 2')
    # ax[1].axis('off')  # Hide axes ticks

    # plt.show()
    return np.sum(hit_mask) / np.sum(valid_origin)


def find_central_image_multi_thread(depth_file_paths, poses, intrinsics):
    """Find the central image based on maximum overlap using multithreading."""
    num_images = len(depth_file_paths)
    overlap_matrix = np.zeros((num_images, num_images))

    # Function to load and convert depth images
    def load_and_convert_image(path):
        return np.array(Image.open(path)) / 1000.0  # Convert to meters

    # Load depth images with multithreading
    with ThreadPoolExecutor() as executor:
        depth_images = list(executor.map(load_and_convert_image, depth_file_paths))
    print(f'done load {len(depth_images)} depth images')
    # Convert depth images to point clouds
    point_clouds = [depth_image_to_point_cloud(img, intrinsics) for img in depth_images]

    # Helper function to compute overlap for a pair of images
    def compute_overlap_for_pair(i, j):
        if i == j:
            return 1
        relative_pose = compute_relative_pose(poses[i], poses[j])
        transformed_points = transform_point_cloud(point_clouds[j], relative_pose)
        proj_depth_map = project_point_cloud(transformed_points, intrinsics)
        return compute_overlap(depth_images[i], proj_depth_map)

    # Compute overlaps using multithreading
    def process_row(i):
        overlaps = []
        #print(f'process row {i}')
        for j in range(num_images):
            #print(f'{i}-{j}')
            if j < i:
                continue
                #overlaps.append(overlap_matrix[j][i])  # Use symmetry property
            else:
                overlap = compute_overlap_for_pair(i, j)
                #overlaps.append(overlap)
                overlap_matrix[j][i] = overlap  # Fill symmetry
                overlap_matrix[i][j] = overlap 
        return overlaps

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(process_row, range(num_images)))
        #for i, row in enumerate(results):
        #    print(overlap_matrix[i])

    # Find the image with the highest average overlap
    average_overlaps = np.mean(overlap_matrix, axis=1)
    central_idx = np.argmax(average_overlaps)

    # Identify bad images with overlap against the central image below the threshold
    threshold = np.quantile(overlap_matrix[central_idx], 0.03) #0.081#np.quantile(overlap_matrix[central_idx], 0.03)
    bad_images_idxs = [idx for idx, overlap in enumerate(overlap_matrix[central_idx]) if overlap < threshold]

    return central_idx, bad_images_idxs



def compute_pose_difference(pose1, pose2):
    '''
        pose tx, ty, tz, qx, qy, qz, qw 
    '''
    r1, t1 = R.from_quat(pose1[3:]), np.array(pose1[:3])
    r2, t2 = R.from_quat(pose2[3:]), np.array(pose2[:3])
    rotation_diff = R.inv(r1) * r2
    translation_diff = np.linalg.norm(t1 - t2)
    return  translation_diff, rotation_diff.magnitude()

def find_central_pose(poses):
    """ Find the pose with the smallest average distance to all other poses. 
        pose tx, ty, tz, qx, qy, qz, qw 
    """
    min_avg_distance = float('inf')
    central_pose = None
    central_idx = 0

    for i, pose1 in enumerate(poses):
        total_distance = sum(compute_pose_difference(pose1, pose2)[0] for j, pose2 in enumerate(poses) if i != j)
        avg_distance = total_distance / (len(poses) - 1)

        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            central_pose = pose1
            central_idx = i

    return central_idx, central_pose


# if relative is True, saved posed should be relative to the first pose
def save_intrinsics_and_poses_mfl(output_dir, frame_paths, intrinsics, poses, relative=True):
    with open(os.path.join(output_dir, 'intrinsics.txt'), 'w') as f:
        for frame_path in frame_paths:
            f.write(f"{frame_path} {' '.join(map(str, intrinsics))}\n")
            
    # Get the first pose and compute its inverse
    first_pose = poses[0]
    print('output first pose ', first_pose)
    tx, ty, tz, qx, qy, qz, qw = first_pose
    first_rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
    first_rotation_matrix_inv = first_rotation_matrix.T
    first_translation_vector_inv = -first_rotation_matrix_inv @ np.array([tx, ty, tz])
    os.makedirs(os.path.join(output_dir, 'fake_estimation'), exist_ok=True)
    with open(os.path.join(output_dir, 'fake_estimation/poses.txt'), 'w') as est_f:
        with open(os.path.join(output_dir, 'poses.txt'), 'w') as f:
            with open(os.path.join(output_dir, 'pose_abs.txt'), 'w') as tum_f:
                for frame_path, pose  in zip(frame_paths, poses ):
                    tx, ty, tz, qx, qy, qz, qw = pose
                    rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
                    translation_vector = np.array([tx, ty, tz])
                    if relative:
                        # Compute the relative rotation and translation to the first pose
                        relative_rotation_matrix = first_rotation_matrix_inv @ rotation_matrix
                        relative_translation_vector = first_rotation_matrix_inv @ translation_vector + first_translation_vector_inv
                        rotation_matrix = relative_rotation_matrix
                        translation_vector = relative_translation_vector
                    
                    rotation_matrix_inv = rotation_matrix.T
                    translation_vector_inv = -rotation_matrix_inv @ translation_vector
                    inverted_quat = R.from_matrix(rotation_matrix_inv).as_quat()
            
                    inverted_pose = np.roll(inverted_quat, 1).tolist() + translation_vector_inv.tolist() 
                    f.write(f"{frame_path} {' '.join(map(str, inverted_pose))}\n")
                    # fake estimation file
                    est_f.write(f"{frame_path} {' '.join(map(str, inverted_pose+[1]))}\n")
                    # Write to TUM pose file
                    tum_f.write(f"{frame_path} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")


def compare_image_lists(dir1, dir2):
    images1 = set(os.listdir(dir1))
    images2 = set(os.listdir(dir2))
    intersection = images1.intersection(images2)
    smaller_set_size = min(len(images1), len(images2))
    return (len(intersection) / smaller_set_size) >= 0.8 if smaller_set_size != 0 else False

def find_similar_regions(directories):
    grouped = []
    used = set()

    for dir1 in directories:
        if dir1 in used:
            continue
        current_group = [dir1]
        used.add(dir1)
        for dir2 in directories:
            if dir2 not in used and compare_image_lists(os.path.join(dir1, 'images'), os.path.join(dir2, 'images')):
                current_group.append(dir2)
                used.add(dir2)
        grouped.append(current_group)
    
    return grouped

def get_center_position(region_stat_path):
    with open(region_stat_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "Center" in line:
                center_part = line.split('=')[1].strip()
                center = np.array(center_part.strip('[]').split(), dtype=float)
                return center
    return None

def find_close_regions(directories):
    grouped = []
    used = set()
    centers = {}

    # Pre-read centers
    for dir in directories:
        stat_path = os.path.join(dir, 'region_stat.txt')
        centers[dir] = get_center_position(stat_path)

    for dir1 in directories:
        if dir1 in used:
            continue
        current_group = [dir1]
        used.add(dir1)
        center1 = centers[dir1]
        for dir2 in directories:
            if dir2 not in used:
                center2 = centers[dir2]
                if np.linalg.norm(center1 - center2) < 25:  # Using Euclidean distance to compare
                    current_group.append(dir2)
                    used.add(dir2)
        grouped.append(current_group)
    
    return grouped

def adjust_indices(central_idx, bad_indices):
    """Adjust bad indices after the central index has been moved to the front."""
    adjusted_indices = []
    for idx in bad_indices:
        if idx > central_idx:
            adjusted_indices.append(idx - 1)
        elif idx < central_idx:
            adjusted_indices.append(idx + 1)
    return adjusted_indices

def is_pose_similar(pose1, pose2, th_t = 1, th_r=2*math.pi/180):
    d_t, d_r = compute_pose_difference(pose1, pose2) 
    #print (pose1, pose2, d_t, d_r)
    return d_t<th_t and d_r < th_r

def process_region_list(region_dirs, output_dir):
    img_file_paths = []
    img_positions = []
    img_rotations = []

    # Aggregate data from all directories
    for region_dir in region_dirs:
        json_path = os.path.join(region_dir, 'region_data.json')
        img_dir = os.path.join(region_dir, 'images')

        # Load JSON data
        with open(json_path, 'r') as file:
            data = json.load(file)

        # Convert to local img paths and aggregate pose data
        for index, file_path in enumerate(data['img_file_paths']):
            parts = file_path.split('/')[-4:]  # Get the last three parts: scene, seq, image_dir, filename
            parts.pop(-2)
            new_file_name = '_'.join(parts)
            new_file_path = os.path.join(img_dir, new_file_name)
            new_pose = data['img_positions'][index] + data['img_rotations'][index]
            #
            # Check for duplicates in paths and poses
            is_duplicate_path = new_file_path in img_file_paths
            is_duplicate_pose = any(is_pose_similar(new_pose, pose) for pose in (p + r for p, r in zip(img_positions, img_rotations)))

            if not is_duplicate_path and not  is_duplicate_pose:
                img_file_paths.append(new_file_path)
                img_positions.append(data['img_positions'][index])
                img_rotations.append(data['img_rotations'][index])
            # if new_file_path not in img_file_paths:
            #     img_file_paths.append(new_file_path)
            #     img_positions.append(data['img_positions'][index])
            #     img_rotations.append(data['img_rotations'][index])
            #img_file_paths.append(new_file_path)

        #img_positions.extend(data['img_positions'])
        #img_rotations.extend(data['img_rotations'])

    # Continue with the rest of the processing
    img_poses = [position + rotation for position, rotation in zip(img_positions, img_rotations)]
    intrinsics = (913.896, 912.277, 638.954, 364.884, 1280, 720)  # Example intrinsics, adjust as needed

    print(f'Current region group with {len(region_dirs)} regions {len(img_poses)} images')
    # find central pose by overlap
    depth_file_paths = [filename.replace('.jpg', '.ray_neighbor.png') for filename in img_file_paths]
    central_idx, bad_images_idxs = find_central_image_multi_thread(depth_file_paths, img_poses, intrinsics)
    
    # move ref to front
    closest_image_path = img_file_paths.pop(central_idx)
    closest_image_pose = img_poses.pop(central_idx)
    img_file_paths.insert(0, closest_image_path)
    img_poses.insert(0, closest_image_pose)
    print('central idx ', central_idx)
    print('central pose ', closest_image_pose)

    # Adjust bad indices since we've moved the central item
    bad_images_idxs = adjust_indices(central_idx, bad_images_idxs)
    print(output_dir, ' bad idxs after ', bad_images_idxs)
    # Remove bad images and poses
    for idx in sorted(bad_images_idxs, reverse=True):
        del img_file_paths[idx]
        del img_poses[idx]
    print(f'remove {len(bad_images_idxs)} bad images')

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    seq0_dir = os.path.join(output_dir, 'seq0')
    os.makedirs(seq0_dir, exist_ok=True)
    seq1_dir = os.path.join(output_dir, 'seq1')
    os.makedirs(seq1_dir, exist_ok=True)

    # Move and rename the closest image and its depth image
    src_img_path = closest_image_path
    dst_img_path = os.path.join(seq0_dir, 'frame_00000.jpg')
    shutil.copy(src_img_path, dst_img_path)
    destination_paths = [dst_img_path]

    # Handle depth image similarly
    closest_depth_path = closest_image_path.replace('.jpg', '.ray_neighbor.png')
    src_depth_path = closest_depth_path
    dst_depth_path = os.path.join(seq0_dir, 'frame_00000.ray_neighbor.png')
    shutil.copy(src_depth_path, dst_depth_path)

    for idx, img_path in enumerate(img_file_paths):
        if idx == 0:
            continue
        src_img_path = img_file_paths[idx]
        dst_img_path = os.path.join(seq1_dir, f'frame_{(idx-1):05d}.jpg')
        shutil.copy(src_img_path, dst_img_path)
        destination_paths.append(dst_img_path)
   
        src_depth_path = src_img_path.replace('.jpg', '.ray_neighbor.png')
        dst_depth_path = os.path.join(seq1_dir, f'frame_{(idx-1):05d}.ray_neighbor.png')
        shutil.copy(src_depth_path, dst_depth_path)

    # Adjust paths to be relative to the scene directory
    all_frame_paths = [os.path.join('seq0', os.path.basename(fp)) if 'seq0' in fp else os.path.join('seq1', os.path.basename(fp)) for fp in destination_paths]
    
    save_intrinsics_and_poses_mfl(output_dir, all_frame_paths, intrinsics, img_poses, relative=True) 

# Example usage:
base_dir = '/media/host/OneTouch/APMP_dataset'
input_root = os.path.join(base_dir, 'region_selection_ready')
output_root = os.path.join(base_dir, 'region_selection_mfl_format_doublecheck_1')


# Iterate over all items in input_root and merge similar 
directories = [os.path.join(input_root, d) for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
similar_regions = find_close_regions(directories)
print (f'found {len(similar_regions)} regions')
print([r for r in similar_regions if len(r) > 0])
for index, region_group in enumerate(similar_regions):
    output_dir = os.path.join(output_root, f's{index:05d}')
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
   # if index = 4:
    process_region_list(region_group, output_dir)
