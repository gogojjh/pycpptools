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
│   ├── pcd/                           # [input] Directory for point cloud data (.pcd files)
│   │   └── [stamp_ns].pcd             # Point cloud files named by timestamp in nanoseconds
│   │
│   ├── images_sampled/                # [input] Directory for processed or sampled images
│   │   └── [stamp_ns].jpg             # Sampled images, named by timestamp
│   │
│   ├── images_raw/                    # [no need] Directory for raw images
│   │   └── [stamp_ns].jpg             # Raw images, named by timestamp
│   │
│   ├── lidar_ref.txt                  # [no need] Text file, references for LiDAR data
│   ├── images_tum.txt                 # [no need] Text file with image data in TUM format (potentially timestamps and filenames)
│   ├── images_sampled_tum.txt         # [output] Similar to images_tum.txt but for the sampled images
│   └── xxx.bag                        # [no need] ROS bag file containing raw data for the sequence


'''

import numpy as np
import os
import shutil
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
import random

def parse_tum_file(tum_file_path):
    data = {}
    timestamps = []
    with open(tum_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 8:
                stamp_s, x, y, z, qx, qy, qz, qw = parts
                stamp_ns = int(float(stamp_s) * 1e9)
                data[stamp_ns] = {
                    'position': np.array([float(x), float(y), float(z)]),
                    'rotation': [float(qx), float(qy), float(qz), float(qw)]
                }
                timestamps.append(stamp_ns)
    return data, np.array(timestamps)

def find_closest_timestamp(target, timestamps):
    idx = np.abs(timestamps - target).argmin()
    return timestamps[idx]

def load_data(seq_path):
    data = {'lidar': {}, 'image': {}}
    lidar_data, lidar_timestamps = parse_tum_file(os.path.join(seq_path, 'lidar_ref.txt'))
    lidar_path = os.path.join(seq_path, 'pcd')
    for pcd_file in os.listdir(lidar_path):
        if pcd_file.endswith('.pcd'):
            stamp_ns = int(pcd_file.split('.')[0])
            closest_stamp = find_closest_timestamp(stamp_ns, lidar_timestamps)
            data['lidar'][stamp_ns] = lidar_data[closest_stamp]
            data['lidar'][stamp_ns]['file_path'] = os.path.join(lidar_path, pcd_file)

    image_data, image_timestamps = parse_tum_file(os.path.join(seq_path, 'images_sampled_tum.txt'))
    images_path = os.path.join(seq_path, 'images_sampled')
    for img_file in os.listdir(images_path):
        if img_file.endswith('.jpg'):
            stamp_ns = int(img_file.split('.')[0])
            closest_stamp = find_closest_timestamp(stamp_ns, image_timestamps)
            data['image'][stamp_ns] = image_data[closest_stamp]
            data['image'][stamp_ns]['file_path'] = os.path.join(images_path, img_file)
    
    return data

def perform_global_kmeans_clustering(all_positions, n_clusters=100):
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_positions)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    return labels, centers

def perform_global_meanshift_clustering(all_positions, quantile=0.02, n_samples=None):
    # Estimate the bandwidth to use with the Mean Shift algorithm
    bandwidth = estimate_bandwidth(all_positions, quantile=quantile, n_samples=n_samples)

    # Perform Mean Shift clustering
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    meanshift.fit(all_positions)
    centers = meanshift.cluster_centers_
    labels = meanshift.labels_

    return labels, centers

def generate_grid_centers(all_positions):
    # Convert coordinates to a numpy array if it's not already
    coordinates = np.array(all_positions)
    
    # Compute min and max bounds
    min_x, min_y, min_z = np.min(coordinates, axis=0)
    max_x, max_y, max_z = np.max(coordinates, axis=0)
    
    # Create mesh grid of points every 25 meters within the bounds
    x_points = np.arange(min_x, max_x, 25)
    y_points = np.arange(min_y, max_y, 25)
    grid_x, grid_y = np.meshgrid(x_points, y_points)
    
    # Flatten the meshgrid arrays and combine them into a list of coordinate tuples
    grid_centers = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    return grid_centers

def generate_occupied_grid_centers(all_positions, grid_size=25, grid_size_z=3):
    # Convert coordinates to a numpy array if it's not already
    coordinates = np.array(all_positions)
    
    # Compute min and max bounds
    min_x, min_y, min_z = np.min(coordinates, axis=0)
    max_x, max_y, max_z = np.max(coordinates, axis=0)
    
    # Create mesh grid of points every 'grid_size' meters within the bounds
    x_points = np.arange(min_x, max_x + grid_size, grid_size)
    y_points = np.arange(min_y, max_y + grid_size, grid_size)
    z_points = np.arange(min_z, max_z + grid_size_z, grid_size_z)
    grid_x, grid_y, grid_z = np.meshgrid(x_points, y_points, z_points, indexing='ij')
    
    # Flatten the meshgrid arrays and combine them into a list of coordinate tuples
    grid_centers = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    
    # Create a boolean array to keep track of occupied cells
    occupied = np.zeros(len(grid_centers), dtype=bool)
    
    # Determine which cells contain points
    for point in coordinates:
        index_x = int((point[0] - min_x) // grid_size)
        index_y = int((point[1] - min_y) // grid_size)
        index_z = int((point[2] - min_z) // grid_size_z)
        linear_index = (index_x * len(y_points) * len(z_points)) + (index_y * len(z_points)) + index_z
        if linear_index < len(occupied):
            occupied[linear_index] = True
    
    # Filter grid centers to include only those that are occupied
    occupied_grid_centers = grid_centers[occupied]
    
    return occupied_grid_centers

def calculate_distances(positions, center):
    distances = cdist(positions, [center])
    min_distance = distances.min()
    max_distance = distances.max()
    avg_distance = distances.mean()
    return min_distance, max_distance, avg_distance

def quaternion_to_direction_vector(qx, qy, qz, qw):
    # Create a Rotation object from the quaternion
    rotation = R.from_quat([qx, qy, qz, qw])
    # Assuming the camera's forward vector points along the z-axis in its local frame
    forward_vector = [0, 0, 1]
    # Apply rotation to the forward vector
    direction_vector = rotation.apply(forward_vector)
    return direction_vector

def is_facing_towards(position, quaternion, target, threshold_angle=np.radians(30)):
    
    # Calculate camera facing direction vector
    camera_direction = quaternion_to_direction_vector(*quaternion)
    
    # Calculate vector from camera to target
    vector_to_target = target - position
    vector_to_target /= np.linalg.norm(vector_to_target)  # Normalize it

    # Calculate the dot product
    dot_product = np.dot(camera_direction, vector_to_target)

    # Calculate the angle between the two vectors
    angle = np.arccos(dot_product)

    # Check if the angle is within the threshold
    return angle <= threshold_angle

def shuffle_and_pick(region_data):
    # Shuffle and limit file paths and analyze scenes and sequences
    for idx, data in region_data.items():
        img_file_paths = data['img_file_paths']
        img_positions = data['img_positions']
        img_rotations = data['img_rotations']
        
        # Check if there are more than 100 image file paths
        if len(img_file_paths) > 100:
            # Combine the lists into a single list of tuples
            combined = list(zip(img_file_paths, img_positions, img_rotations))
            
            # Shuffle the combined list
            random.shuffle(combined)
            
            # Unzip the combined list back into separate lists
            shuffled_img_file_paths, shuffled_img_positions, shuffled_img_rotations = zip(*combined)
            
            # Take the first 100 elements from the shuffled lists
            img_file_paths = list(shuffled_img_file_paths[:100])
            img_positions = list(shuffled_img_positions[:100])
            img_rotations = list(shuffled_img_rotations[:100])
            
            # Update the region data
            data['img_file_paths'] = img_file_paths
            data['img_positions'] = img_positions
            data['img_rotations'] = img_rotations

        # Initialize 'for_mapping' key if it doesn't exist
        if 'for_mapping' not in region_data[idx]:
            region_data[idx]['for_mapping'] = []

        # Analyze scene and sequence importance
        scene_seq_count = {}
        for path in img_file_paths:
            parts = path.split('/')[-4:]
            scene = parts[0]
            seq = parts[1]
            scene_seq = f"{scene}-{seq}"
            if scene_seq in scene_seq_count:
                scene_seq_count[scene_seq] += 1
            else:
                scene_seq_count[scene_seq] = 1

        # Find the most important sequence for each scene
        important_seq_per_scene = {}
        for scene_seq, count in scene_seq_count.items():
            scene, seq = scene_seq.split('-')
            if scene not in important_seq_per_scene or scene_seq_count[scene_seq] > scene_seq_count[important_seq_per_scene[scene]]:
                important_seq_per_scene[scene] = scene_seq

        # Append most important sequences to 'for_mapping'
        for scene_seq in important_seq_per_scene.values():
            if scene_seq not in region_data[idx]['for_mapping']:
                region_data[idx]['for_mapping'].append(scene_seq)
    return region_data    


def convert_numpy(obj):
    """
    Recursively convert numpy data types to python native data types for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert numpy numbers to python scalars
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}  # Apply recursively for dictionaries
    elif isinstance(obj, list):
        return [convert_numpy(item) for item in obj]  # Apply recursively for lists
    return obj

def dump_region_data_to_file(region_data, file_path):
    import json
    """
    Dumps the region data to a JSON file.
    
    Args:
    region_data (dict): The data to dump.
    file_path (str): The path to the file where the data should be written.
    """
    try:
        # Convert numpy objects to native Python types
        region_data_converted = convert_numpy(region_data)
        with open(file_path, 'w') as file:
            json.dump(region_data_converted, file, indent=4)
        print(f"Data successfully written to {file_path}")
    except Exception as e:
        print(f"An error occurred while writing data to {file_path}: {e}")

def process_scenes():
    base_dir = '/media/host/OneTouch/APMP_dataset'
    scenes_to_process = ['cainiao', 'huoniao', 'fantang', 'tiyuguan', 'lab' ]
    all_lidar_positions = []
    lidar_info = []
    all_image_positions = []
    image_info = []

    # Collect all positions
    for scene in scenes_to_process:
        scene_dir = os.path.join(base_dir, scene)
        seqnames = [d for d in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, d))]
        for seqname in seqnames:
            seq_path = os.path.join(scene_dir, seqname)
            data = load_data(seq_path)
            for key, value in data['lidar'].items():
                all_lidar_positions.append(value['position'])
                lidar_info.append((value['position'],value['rotation'], value['file_path']))
            for key, value in data['image'].items():
                all_image_positions.append(value['position'])
                image_info.append((value['position'],value['rotation'], value['file_path']))

    # Perform global k-means clustering
    # image_labels, image_centers = perform_global_kmeans_clustering(np.array(all_image_positions))
    #image_labels, image_centers = perform_global_meanshift_clustering(np.array(all_image_positions))
    image_centers = generate_occupied_grid_centers(np.array(all_image_positions))

    region_data = {i: {'img_positions': [],'img_rotations': [],'lidar_positions': [],'lidar_rotations': [],  'img_file_paths': [], 'lidar_file_paths': []} for i in range(len(image_centers))}

    # Assign file paths to clusters and calculate distances
    # for (position, file_path), label in zip(image_info, image_labels):
    #     region_data[label]['positions'].append(position)
    #     region_data[label]['img_file_paths'].append(file_path)

    #Assign images to clusters by radius
    distance_threshold = 30
    for (position, rotation, file_path) in image_info:
        distances = np.linalg.norm(image_centers - position, axis=1)
        # closest_idx = np.argmin(distances)
        # if distances[closest_idx] < distance_threshold:
        #     region_data[closest_idx]['lidar_file_paths'].append(file_path)
        for idx, distance in enumerate(distances):
            if distance < distance_threshold:
                # Check if the camera is facing towards the region center
                if is_facing_towards(position, rotation, image_centers[idx]):
                    region_data[idx]['img_positions'].append(position)
                    region_data[idx]['img_rotations'].append(rotation)
                    region_data[idx]['img_file_paths'].append(file_path)
    
    region_data = shuffle_and_pick(region_data)

    # Assign lidar to image clusters by radius
    distance_threshold = 30
    for (position, rotation, file_path) in lidar_info:

        distances = np.linalg.norm(image_centers - position, axis=1)
        # closest_idx = np.argmin(distances)
        # if distances[closest_idx] < distance_threshold:
        #     region_data[closest_idx]['lidar_file_paths'].append(file_path)
        parts = file_path.split('/')[-4:]  # Get the last 4 parts: scene, seq, image_dir, filename
        scene_seq_key = f'{parts[0]}-{parts[1]}'
        for idx, distance in enumerate(distances):
            if scene_seq_key not in  region_data[idx]['for_mapping']:
                continue
            if distance < distance_threshold:
                region_data[idx]['lidar_file_paths'].append(file_path)
                region_data[idx]['lidar_positions'].append(position)
                region_data[idx]['lidar_rotations'].append(rotation)

    # New directory for region selections
    region_output_dir = os.path.join(base_dir, "region_selection")
    if not os.path.exists(region_output_dir):
        os.makedirs(region_output_dir)

    for i, data in region_data.items():
        if (len(data['img_file_paths']) < 10):
            continue
        region_dir = os.path.join(region_output_dir, str(i))
        if not os.path.exists(region_dir):
            os.makedirs(region_dir)
        img_dir = os.path.join(region_dir, 'images')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        pcd_dir = os.path.join(region_dir, 'pcd')
        if not os.path.exists(pcd_dir):
            os.makedirs(pcd_dir)

        for file_path in data['img_file_paths']:
            parts = file_path.split('/')[-4:]  # Get the last three parts: scene, seq, image_dir, filename
            parts.pop(-2)
            new_file_name = '_'.join(parts)#.split('.')[0] + '.jpg'
            new_file_path = os.path.join(img_dir, new_file_name)
            shutil.copy2(file_path, new_file_path)
 
        for file_path in data['lidar_file_paths']:
            parts = file_path.split('/')[-4:]  # Get the last three parts: scene, seq, image_dir, filename
            parts.pop(-2)
            new_file_name = '_'.join(parts)#.split('.')[0] + '.pcd'
            new_file_path = os.path.join(pcd_dir, new_file_name)
            shutil.copy2(file_path, new_file_path)

    # Display results and calculate distances
    for i, center in enumerate(image_centers):
        if (len(region_data[i]['img_file_paths']) < 10):
             
            continue
        positions = np.array(region_data[i]['img_positions'])
        min_dist, max_dist, avg_dist = calculate_distances(positions, center)
        print(f"Region {i}: Center = {center}")
        print(f"    Min Distance: {min_dist}")
        print(f"    Max Distance: {max_dist}")
        print(f"    Average Distance: {avg_dist}")
        print(f"    Associated IMAGE Files: {len(region_data[i]['img_file_paths'])}")
        print(f"    Associated LiDAR Files: {len(region_data[i]['lidar_file_paths'])}")
    
        region_dir = os.path.join(region_output_dir, str(i))
        dump_region_data_to_file(region_data[i], os.path.join(region_dir, f'region_data.json'))
        info_path = os.path.join(region_dir, f'region_stat.txt')
        with open(info_path, 'w') as file:
            file.write(f"Region {i}: Center = {center}\n")
            file.write(f"    Min Distance: {min_dist}\n")
            file.write(f"    Max Distance: {max_dist}\n")
            file.write(f"    Average Distance: {avg_dist}\n")
            file.write(f"    Associated IMAGE Files: {len(region_data[i]['img_file_paths'])}\n")
            file.write(f"    Associated LiDAR Files: {len(region_data[i]['lidar_file_paths'])}\n")

process_scenes()
