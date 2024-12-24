#! /usr/bin/env python
import sys
import argparse
import cv2
import numpy as np
import os
from datetime import datetime, timezone
from math import radians, sin, cos, sqrt, atan2
import pymap3d as pm

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
from utils_math.tools_eigen import convert_matrix_to_vec, convert_vec_to_matrix, compute_relative_dis

REF_LATITUDE = 51.538560  # Reference latitude in degrees
REF_LONGITUDE = -0.009759  # Reference longitude in degrees
EDGE_DIS_THRESHOLD = 15.0

def resize_image(image, scale=0.5):
    """Resize the image by the given scale factor."""
    target_width, target_height = 4096, 2048
    resized_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    new_dimensions = (int(resized_image.shape[1] * scale), int(resized_image.shape[0] * scale))
    resized_image = cv2.resize(resized_image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image

def construct_camera_matrix(hfov, vfov, output_size):
    """Construct the camera intrinsic matrix based on FOV and output size."""
    width, height = output_size[1], output_size[0]
    hfov_rad = np.deg2rad(hfov)
    vfov_rad = np.deg2rad(vfov)
    
    fx = (width / 2) / np.tan(hfov_rad / 2)
    fy = (height / 2) / np.tan(vfov_rad / 2)
    cx = width / 2
    cy = height / 2

    camera_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    return camera_matrix

def equirectangular_to_perspective(img, K, R, output_size):
    """Convert equirectangular image to perspective view."""
    width, height = output_size[1], output_size[0]
    # Generate the grid for the perspective image
    i, j = np.meshgrid(np.arange(width), np.arange(height))
    i = i.astype(np.float32)
    j = j.astype(np.float32)
    
    x = (i - K[0,2]) / K[0,0]
    y = (j - K[1,2]) / K[1,1]
    z = np.ones_like(x)

    # Normalize direction vectors
    directions = np.stack([x, y, z], axis=-1)
    norm = np.linalg.norm(directions, axis=2, keepdims=True)
    directions /= norm

    # Apply rotation
    directions_rot = directions @ R.T

    # Convert to spherical coordinates
    theta = np.arctan2(directions_rot[...,0], directions_rot[...,2])
    phi = np.arcsin(directions_rot[...,1])

    # Map to equirectangular image coordinates
    equirect_width = img.shape[1]
    equirect_height = img.shape[0]

    uf = (theta + np.pi) / (2 * np.pi) * equirect_width
    vf = (np.pi/2 + phi) / np.pi * equirect_height

    # Sample the pixels
    perspective = cv2.remap(img, uf.astype(np.float32), vf.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return perspective

def convert_to_enu(latitude, longitude, heading, ref_latitude, ref_longitude, mode='wxyz'):
    """
    Convert latitude, longitude, and heading to ENU (East-North-Up) coordinates
    relative to a reference point using pymap3d.

    Parameters:
    - latitude (float): Latitude in degrees.
    - longitude (float): Longitude in degrees.
    - heading (float): Heading angle in degrees (0 = North, 90 = East, etc.).
    - ref_latitude (float): Reference latitude in degrees.
    - ref_longitude (float): Reference longitude in degrees.

    Returns:
    - e, n, u (float): ENU position relative to the reference point.
    - qw, qx, qy, qz (float): Quaternion representing orientation in ENU.
    """
    # Use pymap3d to convert geodetic to ENU
    e, n, u = pm.geodetic2enu(latitude, longitude, 0, ref_latitude, ref_longitude, 0)
    trans = np.array([e, n, u])

    # Convert heading to radians and adjust for ENU rotation direction
    adjusted_heading = (90 - heading) % 360  # Adjust to counterclockwise rotation
    heading_rad = radians(adjusted_heading)

    # Quaternion representing heading (yaw rotation around Up axis)
    qw = cos(heading_rad / 2)
    qx = 0  # No rotation around East axis
    qy = 0  # No rotation around North axis
    qz = sin(heading_rad / 2)
    if mode == 'wxyz':
        quat = np.array([qw, qx, qy, qz])
    else:
        quat = np.array([qx, qy, qz, qw])

    return trans, quat

def convert_date_to_utc(date):
    """Convert date string (YYYY_MM) to UTC timestamp."""
    year, month = map(int, date.split("_"))
    dt = datetime(year, month, 1, 12, 0, tzinfo=timezone.utc)
    return dt.timestamp()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate perspective views from a panoramic image.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the equirectangular panoramic image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the perspective images.")
    parser.add_argument("--num_clusters", type=int, required=True, help="Number of perspective views to generate.")
    parser.add_argument("--output_size", type=int, nargs=2, required=True, help="Output resolution of perspective images (height, width).")
    parser.add_argument("--hfov", type=float, default=120.0, required=True, help="Horizontal field of view in degrees.")
    parser.add_argument("--vfov", type=float, default=45.0, required=True, help="Vertical field of view in degrees.")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'seq'), exist_ok=True)

    # Construct the camera intrinsic matrix
    camera_matrix = construct_camera_matrix(args.hfov, args.vfov, args.output_size)
    print("Camera Intrinsic Matrix:")
    print(camera_matrix)

    # Data to be stored
    timestamps = np.empty((0, 2), dtype=object)
    intrinsics = np.empty((0, 7), dtype=object)
    gps_data = np.empty((0, 6), dtype=object)
    poses_odom = np.empty((0, 8), dtype=object)
    poses_trans_w2c = np.empty((0, 3), dtype=float)
    edges = np.empty((0, 3), dtype=object)

    # Load each panoramic image
    list_filenames = os.listdir(args.input_path)
    list_filenames.sort()
    cur_ind = 0
    for img_ind, filename in enumerate(list_filenames):
        if 'jpg' not in filename: continue

        img = cv2.imread(os.path.join(args.input_path, filename))
        if img is None:
            print(f"Error: Unable to load image from {args.input_path}")
            return
        print(f"Raw image size: {img.shape[0]}x{img.shape[1]}")

        # Resize the image to half its original size
        img_resized = resize_image(img, scale=1.0)

        # Generate rotation matrices for the number of clusters
        # For simplicity, distribute the views evenly around the Y-axis
        rotation_matrices = []
        angles = np.linspace(-120, 180, args.num_clusters, endpoint=False)
        for angle in angles:
            theta = np.deg2rad(angle)
            R = np.array([
                [ np.cos(theta), 0, np.sin(theta)],
                [             0, 1,           0  ],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
            rotation_matrices.append(R)

        # Generate and save perspective images
        latitude, longitude, date, heading, roll, pitch, pano_id = filename.split('.jpg')[0].split('@')
        print(filename)
        latitude = float(latitude)
        longitude = float(longitude)
        heading = float(heading)
        roll = float(roll)
        pitch = float(pitch)

        tsl_enu, quat_enu = convert_to_enu(latitude, longitude, heading, REF_LATITUDE, REF_LONGITUDE)
        T_enu = convert_vec_to_matrix(tsl_enu, quat_enu)
        T_enu_cam = np.eye(4); T_enu_cam[:3, :3] = np.array([[0,  1,  0], [0,  0, -1], [1,  0,  0]])
        for idx, R in enumerate(rotation_matrices):
            perspective = equirectangular_to_perspective(img_resized, camera_matrix, R, args.output_size)
            T_rot = np.block([[R, np.array([0, 0, 0]).reshape(3, 1)], [0, 0, 0, 1]])
            T_w2c = T_enu @ T_enu_cam @ T_rot

            # Time
            vec = np.empty((1, 2), dtype=object)
            utc_timestamp = convert_date_to_utc(date)
            vec[0, 0], vec[0, 1] = f'seq/{cur_ind:06d}.color.jpg', utc_timestamp
            timestamps = np.vstack((timestamps, vec))

            # GPS
            vec = np.empty((1, 6), dtype=object)
            vec[0, 0], vec[0, 1], vec[0, 2], vec[0, 3], vec[0, 4], vec[0, 5] = \
                f'seq/{cur_ind:06d}.color.jpg', latitude, longitude, np.nan, np.nan, np.nan
            gps_data = np.vstack((gps_data, vec))

            ##### Edges
            if cur_ind > 0:
                dis = np.linalg.norm(T_w2c[:3, 3] - poses_trans_w2c, axis=1)
                dis[np.abs(dis) <= 1e-6] = 1e-6
                indices = np.where(dis <= EDGE_DIS_THRESHOLD)
                for ind in indices[0]:
                    edges = np.vstack((edges, np.array([ind, cur_ind, dis[ind]])))

            # Poses
            T_c2w = np.linalg.inv(T_w2c)
            tsl, quat = convert_matrix_to_vec(T_c2w)
            vec = np.empty((1, 8), dtype=object)
            vec[0, 0], vec[0, 1:5], vec[0, 5:] = f'seq/{cur_ind:06d}.color.jpg', quat, tsl
            poses_odom = np.vstack((poses_odom, vec))
            poses_trans_w2c = np.vstack((poses_trans_w2c, T_w2c[:3, 3].reshape(1, 3)))

            # Intrinsics
            vec = np.empty((1, 7), dtype=object)
            vec[0, 0], vec[0, 1], vec[0, 2], vec[0, 3], vec[0, 4], vec[0, 5], vec[0, 6] = \
                f'seq/{cur_ind:06d}.color.jpg', \
                camera_matrix[0, 0], camera_matrix[1, 1], \
                camera_matrix[0, 2], camera_matrix[1, 2], \
                perspective.shape[1], perspective.shape[0]
            intrinsics = np.vstack((intrinsics, vec))
            
            # Image
            output_img_path = os.path.join(args.output_path, 'seq', f"{cur_ind:06d}.color.jpg")
            cv2.imwrite(output_img_path, perspective)
            cur_ind += 1

    np.savetxt(os.path.join(args.output_path, 'timestamps.txt'), timestamps, fmt='%s %.9f')
    np.savetxt(os.path.join(args.output_path, 'intrinsics.txt'), intrinsics, fmt='%s' + ' %.6f' * 4 + ' %d' * 2)
    np.savetxt(os.path.join(args.output_path, 'gps_data.txt'), gps_data, fmt='%s' + ' %.6f' * 5)
    np.savetxt(os.path.join(args.output_path, 'poses.txt'), poses_odom, fmt='%s' + ' %.6f' * 7)
    np.savetxt(os.path.join(args.output_path, 'odometry_edge_list.txt'), edges, fmt='%d %d %.6f')

if __name__ == "__main__":
    main()
