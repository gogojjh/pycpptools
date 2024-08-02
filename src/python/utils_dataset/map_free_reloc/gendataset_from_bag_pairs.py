'''
Author: David-Willo davidwillo@foxmail.com
Date: 2024-08-02 05:42:28
LastEditTime: 2024-08-02 07:50:37
LastEditors: David-Willo
Jinhao HE (David Willo), IADC HKUST(GZ)
Copyright (c) 2024 by davidwillo@foxmail.com, All Rights Reserved. 
'''
"""
Author: Jinhao He
Date: 2024-08-01
Description: This script generates data consistent with map-free-reloc format from rosbag.
Source: https://github.com/nianticlabs/map-free-reloc
Version: 1.1
"""

"""Format of Map-Free-Reloc
s004600/
    seq0/
        frame_00000.png
        frame_00000.xxx.png (xxx indicates the depth estimation method)
    seq1/
        frame_00000.png
        frame_00000.xxx.png
        frame_00001.png
        frame_00001.xxx.png
        ...
    poses.txt (format: image_path qw qx qy qz tx ty tz)
    intrinsics.txt (format: image_path fx fy cx cy width height)
    overlaps.npz (idxs.npy [[seqid frameid seqid frameid]] overlaps.npy [scores])
"""



import os
import numpy as np
import yaml
import rosbag
from cv_bridge import CvBridge
import cv2
import tf
import math
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from matching import get_matcher

device = 'cuda'  # or 'cpu'
matcher = get_matcher('"xfeat-lg', device=device)  # Choose any of our ~20+ matchers listed below
img_size = 640

def read_lidar_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            timestamp = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            poses.append((timestamp, np.array([tx, ty, tz, qx, qy, qz, qw])))
    return poses

def interpolate_pose(poses, timestamp):
    poses = sorted(poses, key=lambda x: x[0])
    if timestamp <= poses[0][0]:
        return poses[0][1]
    if timestamp >= poses[-1][0]:
        return poses[-1][1]
    for i in range(len(poses) - 1):
        if poses[i][0] <= timestamp <= poses[i + 1][0]:
            t0, pose0 = poses[i]
            t1, pose1 = poses[i + 1]
            alpha = (timestamp - t0) / (t1 - t0)
            trans0, trans1 = pose0[:3], pose1[:3]
            quat0, quat1 = pose0[3:], pose1[3:]
            trans = (1 - alpha) * trans0 + alpha * trans1
            quat = tf.transformations.quaternion_slerp(quat0, quat1, alpha)
            return  np.hstack((trans, quat)).tolist()
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

def compute_pose_difference(pose1, pose2):
    r1, t1 = R.from_quat(pose1[3:]), np.array(pose1[:3])
    r2, t2 = R.from_quat(pose2[3:]), np.array(pose2[:3])
    rotation_diff = R.inv(r1) * r2
    translation_diff = np.linalg.norm(t1 - t2)
    return  translation_diff, rotation_diff.magnitude()

def generate_keyframes(poses, trans_thresh=15.0, rot_thresh=5 * (math.pi / 180.0)):
    keyframes = [poses[0]]
    
    for pose in poses[1:]:
        doadd = True
        #print(pose)
        #print(keyframes[-1])
        trans_diff, rot_diff = compute_pose_difference(pose, keyframes[-1])
        if trans_diff < trans_thresh :#or rot_diff > rot_thresh:
            continue
        
        for kf in keyframes:
            trans_diff, rot_diff = compute_pose_difference(pose, kf)
            #print(trans_thresh, rot_thresh, trans_diff,rot_diff)
        #doadd = False
            if trans_diff < trans_thresh and rot_diff < rot_thresh :
                doadd = False
        if doadd:    
            keyframes.append(pose)
    return keyframes
  
def extract_images_and_save(bag_file, output_dir, topic, intrinsics, distortion, skip=False):
    os.makedirs(output_dir, exist_ok=True)
    bag = rosbag.Bag(bag_file, 'r')
    bridge = CvBridge()
    frame_paths = []
    timestamps = []
    fx, fy, cx, cy, width, height = intrinsics
    k1, k2, p1, p2 = distortion
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    D = np.array([k1, k2, 0, p1, p2])
    
    # Precompute the undistortion and rectification maps
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (width, height), cv2.CV_32FC1)
    for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[topic])):
        frame_path = os.path.join(output_dir, f'frame_{i:05d}.jpg')
        frame_paths.append(frame_path)
        timestamps.append(t.to_sec())
        if not skip:
            cv_image = bridge.compressed_imgmsg_to_cv2(msg)
            resized_image = cv2.resize(cv_image, (width, height))
            #undistorted_image = cv2.undistort(resized_image, K, D, None, K)
            undistorted_image = cv2.remap(resized_image, map1, map2, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(frame_path, undistorted_image)
    bag.close()
    return frame_paths, timestamps

# if relative is True, saved posed should be relative to the first pose
def save_intrinsics_and_poses(output_dir, frame_paths, intrinsics, poses, relative=True):
    with open(os.path.join(output_dir, 'intrinsics.txt'), 'w') as f:
        for frame_path in frame_paths:
            f.write(f"{frame_path} {' '.join(map(str, intrinsics))}\n")
            
    # Get the first pose and compute its inverse
    first_pose = poses[0]
    tx, ty, tz, qx, qy, qz, qw = first_pose
    first_rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
    first_rotation_matrix_inv = first_rotation_matrix.T
    first_translation_vector_inv = -first_rotation_matrix_inv @ np.array([tx, ty, tz])
    
    with open(os.path.join(output_dir, 'poses.txt'), 'w') as f:
        for frame_path, pose in zip(frame_paths, poses):
            tx, ty, tz, qx, qy, qz, qw = pose
            rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
            translation_vector = np.array([tx, ty, tz])
            if relative:
                # Compute the relative rotation and translation to the first pose
                relative_rotation_matrix = first_rotation_matrix_inv @ rotation_matrix
                relative_translation_vector = first_rotation_matrix_inv @ (translation_vector - first_translation_vector_inv)
                rotation_matrix = relative_rotation_matrix
                translation_vector = relative_translation_vector
            
            rotation_matrix_inv = rotation_matrix.T
            translation_vector_inv = -rotation_matrix_inv @ translation_vector
            inverted_quat = R.from_matrix(rotation_matrix_inv).as_quat()
     
            inverted_pose = np.roll(inverted_quat, 1).tolist() + translation_vector_inv.tolist()
            f.write(f"{frame_path} {' '.join(map(str, inverted_pose))}\n")


def is_pose_similar(pose1, pose2, trans_thresh=0.1, rotation_threshold=2*math.pi/180):
    trans_diff_neighbor, rot_diff_neighbor = compute_pose_difference(pose1, pose2)
    if trans_diff_neighbor < trans_thresh and rot_diff_neighbor < rotation_threshold:
        return True
    return False

def find_closest_keyframe(ref_keyframes, target_keyframes, trans_thresh=15, rotation_threshold=30*math.pi/180):
    ref_kdtree = KDTree([pose[:3] for pose in ref_keyframes])
    closest_keyframes = []
    for target_pose in target_keyframes:
        # Query the KDTree to find the indices of the top 3 closest keyframe
        dists, indices = ref_kdtree.query(target_pose[:3], k=5)
        
        for idx in indices:
            ref_pose = ref_keyframes[idx]
            # Check if the poses are similar
            if is_pose_similar(target_pose, ref_pose, trans_thresh, rotation_threshold):
                closest_keyframes.append(ref_pose)
                break  # Stop after finding the first satisfying keyframe
            # DW: actually should add None even if cannot find correspondence
    return closest_keyframes
    
def find_neighbors(frame_paths, camera_poses, keyframe_pose, trans_thresh_keyframes, rotation_threshold):
    neighbors = []
    keyframe_index = None
    min_diff_t, min_diff_rot = 999, 2*math.pi
    # the closest is the keyframe, move the keyframe to the font of the list, the first element should be reference
    
    for j in range(len(frame_paths)):
        trans_diff, rot_diff = compute_pose_difference(camera_poses[j], keyframe_pose)
        if trans_diff< trans_thresh_keyframes and rot_diff < rotation_threshold:
            should_skip = False
            for neighbor in neighbors:
                # TODO: is this efficient?
                if is_pose_similar(camera_poses[j], camera_poses[frame_paths.index(neighbor)]):
                    should_skip = True
                    break
            if not should_skip:
                neighbors.append(frame_paths[j])
                if trans_diff < min_diff_t :
                    min_diff_t = trans_diff
                    keyframe_index = len(neighbors) - 1
                
    if keyframe_index is not None:
        keyframe = neighbors.pop(keyframe_index)
        neighbors.insert(0, keyframe)
    else:
        raise
    return neighbors
    
def compute_covisibility_score_naive(image_pair):
    imageA, imageB = image_pair
    # Convert images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect and compute SIFT keypoints and descriptors
    keypointsA, descriptorsA = sift.detectAndCompute(grayA, None)
    keypointsB, descriptorsB = sift.detectAndCompute(grayB, None)

    # Initialize the Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptorsA, descriptorsB)

    # Calculate the covisibility score
    if len(keypointsA) == 0:
        return 0.0
    score = len(matches) / len(keypointsA)
    return score

def compute_covisibility_score(image_pair):
    img0, img1 = image_pair

    result = matcher(img0, img1)
    num_inliers = result['num_inliers']
    total_keypoints = len(result['kpts0'])

    if total_keypoints == 0:
        return 0.0
    score = num_inliers / total_keypoints
    return score

def load_images_cv2(frame_paths):
    images = []
    for path in tqdm(frame_paths, desc="Loading images"):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        images.append(img)
    return images

def load_image(path):
    return matcher.load_image(path, resize=img_size)


def load_images(frame_paths):
    images = []
    with ThreadPoolExecutor() as executor:
        for img in tqdm(executor.map(load_image, frame_paths), total=len(frame_paths), desc="Loading images"):
            images.append(img)
    return images
    
def compute_pairwise_covisibility(all_frame_paths, seq0_dir, seq1_dir, scene_dir):
    # Set multiprocessing start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    num_images = len(all_frame_paths)
    images = load_images(all_frame_paths)
    idxs = []
    overlaps = []
    tasks = []
    # with ProcessPoolExecutor(max_workers=1) as executor:
    #     for i in range(num_images):
    #         for j in range(i + 1, num_images):
    #             tasks.append(executor.submit(compute_covisibility_score, (images[i], images[j])))

    #     results = []
    #     for task in tqdm(as_completed(tasks), total=len(tasks), desc="Processing images"):
    #         results.append(task.result())

    tasks = [(images[i], images[j]) for i in range(num_images) for j in range(i + 1, num_images)]

    with ProcessPoolExecutor(max_workers=2) as executor:
        # Use map to submit tasks and collect results
        results = list(tqdm(executor.map(compute_covisibility_score, tasks), total=len(tasks), desc="Processing images"))

    idx = 0
    for i in range(num_images):
        for j in range(i + 1, num_images):
            score = results[idx]
            idx += 1
            # TODO: seqid?
            seqid1 = 0 if all_frame_paths[i].startswith(seq0_dir) else 1
            frameid1 = int(os.path.basename(all_frame_paths[i]).split('_')[1].split('.')[0])
            seqid2 = 0 if all_frame_paths[j].startswith(seq0_dir) else 1
            frameid2 = int(os.path.basename(all_frame_paths[j]).split('_')[1].split('.')[0])

            idxs.append([seqid1, frameid1, seqid2, frameid2])
            overlaps.append(score)

    np.savez(os.path.join(scene_dir, 'overlaps.npz'), idxs=np.array(idxs), overlaps=np.array(overlaps))

def main(bag1, bag2, lidar_ref1, lidar_ref2, camera_extrinsics, output_dir, rot_thresh, trans_thresh, rot_thresh_keyframes, trans_thresh_keyframes, image_topic, skip_images=False, validation_only=False):
    with open(camera_extrinsics, 'r') as f:
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

    # Read lidar poses
    poses1 = read_lidar_poses(lidar_ref1)
    poses2 = read_lidar_poses(lidar_ref2)

    # Extract images and timestamps
    frame_paths1, timestamps1 = extract_images_and_save(bag1, os.path.join(output_dir, 'images1'), image_topic, intrinsics, distortion, skip=skip_images)
    frame_paths2, timestamps2 = extract_images_and_save(bag2, os.path.join(output_dir, 'images2'), image_topic, intrinsics, distortion, skip=skip_images)

    # Interpolate poses for each frame
    interpolate_pose1 = [interpolate_pose(poses1, ts) for ts in timestamps1]
    interpolate_pose2 = [interpolate_pose(poses2, ts) for ts in timestamps2]

    # Convert to camera frame
    camera_poses1 = apply_lidar_camera_extrinsics(interpolate_pose1, R_lc, t_lc)
    camera_poses2 = apply_lidar_camera_extrinsics(interpolate_pose2, R_lc, t_lc)
    print('Done generate camera poses')
    
    # Generate keyframes
    merged_keyframes1 = generate_keyframes(camera_poses1, trans_thresh_keyframes, rot_thresh_keyframes)
    merged_keyframes2 = generate_keyframes(camera_poses2, trans_thresh_keyframes, rot_thresh_keyframes)
    print('Done Generate keyframes {} {}'.format(len(merged_keyframes1), len(merged_keyframes2)))

    # Find closest keyframes 
    # pay attention to the order here
    # closest_keyframes_to_2 = find_closest_keyframe(merged_keyframes1, merged_keyframes2)
    closest_keyframes_to_1 = find_closest_keyframe(merged_keyframes2, merged_keyframes1)
    print('Done Find closest keyframes {}'.format(len(closest_keyframes_to_1)))
    assert len(closest_keyframes_to_1) == len(merged_keyframes1)

    # Output the pose file and copy images
    for i, (kf1, kf2) in enumerate(zip(merged_keyframes1, closest_keyframes_to_1)):
        scene_dir = os.path.join(output_dir, f's{i:05d}')
        os.makedirs(scene_dir, exist_ok=True)

        seq0_dir = os.path.join(scene_dir, 'seq0')
        seq1_dir = os.path.join(scene_dir, 'seq1')
        os.makedirs(seq0_dir, exist_ok=True)
        os.makedirs(seq1_dir, exist_ok=True)

        
        neighbors1 = find_neighbors(frame_paths1, camera_poses1, kf1, trans_thresh, rot_thresh)
        neighbors2 = find_neighbors(frame_paths2, camera_poses2, kf2, trans_thresh, rot_thresh)
        
        
        # NOTE(gogojjh): for validation set, only generate single reference
        if validation_only:
            neighbors1 = neighbors1[:1]
        
        # Copy images and rename starting from 0
        new_frame_paths1 = []
        for idx, neighbor in enumerate(neighbors1):
            new_path = os.path.join(seq0_dir, f'frame_{idx:05d}.jpg')
            os.system(f'cp {neighbor} {new_path}')
            new_frame_paths1.append(new_path)
            

        new_frame_paths2 = []
        for idx, neighbor in enumerate(neighbors2):
            new_path = os.path.join(seq1_dir, f'frame_{idx:05d}.jpg')
            os.system(f'cp {neighbor} {new_path}')
            new_frame_paths2.append(new_path)

        all_frame_paths = new_frame_paths1 + new_frame_paths2
        
        # Compute covisibility score, i think too time sonsuming
        compute_pairwise_covisibility(all_frame_paths, seq0_dir, seq1_dir, scene_dir)
        
        # Adjust paths to be relative to the scene directory
        all_frame_paths = [os.path.join('seq0', os.path.basename(fp)) if 'seq0' in fp else os.path.join('seq1', os.path.basename(fp)) for fp in all_frame_paths]


        all_camera_poses = [camera_pose for camera_pose, frame_path in zip(camera_poses1, frame_paths1) if frame_path in neighbors1] + \
                           [camera_pose for camera_pose, frame_path in zip(camera_poses2, frame_paths2) if frame_path in neighbors2]
                           
        
        save_intrinsics_and_poses(scene_dir, all_frame_paths, intrinsics, all_camera_poses)
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('bag1', help='First ROS bag file')
    parser.add_argument('bag2', help='Second ROS bag file')
    parser.add_argument('lidar_ref1', help='First lidar reference file')
    parser.add_argument('lidar_ref2', help='Second lidar reference file')
    parser.add_argument('camera_extrinsics', help='Camera extrinsics file')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--skip_images', action='store_true', default=False, help='skip extract images from bag')
    parser.add_argument('--validation_only', action='store_true', default=False, help='for validation set only, only consider first bag')
    parser.add_argument('--trans_thresh_keyframes', type=float, default=30.0, help='Distance threshold in meters')
    parser.add_argument('--rot_thresh', type=float, default=40.0, help='Rotation threshold in degrees')
    parser.add_argument('--trans_thresh', type=float, default=5.0, help='Translation threshold in meters')
    parser.add_argument('--rot_thresh_keyframes', type=float, default=5.0, help='Rotation threshold for keyframes in degrees')
    parser.add_argument('--image_topic', default='/usb_cam/image_raw/compressed', help='Image topic')
    args = parser.parse_args()
    
     # Convert degrees to radians for rotation thresholds
    args.rot_thresh *= math.pi / 180.0
    args.rot_thresh_keyframes *= math.pi / 180.0
    main(args.bag1, args.bag2, args.lidar_ref1, args.lidar_ref2, args.camera_extrinsics, args.output_dir, args.rot_thresh, args.trans_thresh, args.rot_thresh_keyframes, args.trans_thresh_keyframes, args.image_topic, args.skip_images, args.validation_only)
