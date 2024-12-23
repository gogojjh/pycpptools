import cv2
import numpy as np
import random

def wrap_panoramic_to_perspective(pano_img, camera_matrix, rotation_matrix, out_shape=(480, 640)):
    """
    Projects an equirectangular panoramic image to a perspective view.
    
    Parameters:
    - pano_img:         Input panoramic image (H_pano, W_pano, 3) or (H_pano, W_pano)
    - camera_matrix:    Camera intrinsic matrix (3, 3)
    - rotation_matrix:  Camera rotation matrix (3, 3)
    - out_shape:        Output image dimensions (height, width)
    
    Returns:
    - persp_img:        Resulting perspective view image
    """
    
    out_h, out_w = out_shape
    pano_h, pano_w = pano_img.shape[:2]
    
    # Create pixel coordinates for the output image
    vv, uu = np.indices((out_h, out_w))
    ones = np.ones_like(uu, dtype=np.float32)
    pixel_coords = np.stack([uu, vv, ones], axis=-1).reshape(-1, 3)  # (N, 3), where N = out_h * out_w

    # Map pixel coordinates to camera coordinates using the intrinsic matrix
    inv_K = np.linalg.inv(camera_matrix)
    cam_coords = pixel_coords @ inv_K.T  # (N, 3)

    # Map camera coordinates to world coordinates using the rotation matrix
    inv_R = np.linalg.inv(rotation_matrix)
    world_coords = cam_coords @ inv_R.T  # (N, 3)

    # Convert 3D world coordinates to spherical coordinates
    x, y, z = world_coords[:, 0], world_coords[:, 1], world_coords[:, 2]
    theta = np.arctan2(y, x)  # Horizontal angle [-pi, pi]
    hypot_xy = np.sqrt(x**2 + y**2)
    phi = np.arctan2(z, hypot_xy)  # Vertical angle [-pi/2, pi/2]

    # Map spherical coordinates to equirectangular image coordinates
    u_equi = (theta + np.pi) / (2 * np.pi) * pano_w
    v_equi = (phi + np.pi / 2) / np.pi * pano_h

    # Reshape and prepare for remapping
    map_x = u_equi.reshape((out_h, out_w)).astype(np.float32)
    map_y = v_equi.reshape((out_h, out_w)).astype(np.float32)

    # Use cv2.remap to sample values from the panoramic image
    persp_img = cv2.remap(
        pano_img,
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP
    )
    
    return persp_img

# Generate a random rotation matrix from random Euler angles
def euler_to_rotmat(rx, ry, rz):
    # Example: rotate in X, then Y, then Z order
    Rx = np.array([
        [1,          0,           0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)]
    ])
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0,           1,         0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1]
    ])
    return Rz @ Ry @ Rx

def main():
    # 1. Read a panoramic image
    pano_path = "/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/raw_vrs/ucl_campus/out_general_streetview/panoramic_image/51.53865535830644@-0.01003246296148459@2017_04@59.394142@359.272644@88.557907@Cq7am7BIw9PZ8diHb2VyrA.jpg"
    pano_img = cv2.imread(pano_path)

    # Resize the panoramic image (optional, based on target width)
    target_width = 6656
    h, w = pano_img.shape[:2]
    ratio = target_width / w
    new_size = (target_width, int(h * ratio))
    pano_img = cv2.resize(pano_img, new_size)

    # Define horizontal and vertical FOV
    horizontal_fov = 120  # degrees
    vertical_fov = 90     # degrees

    # Calculate focal length based on FOV
    out_shape = (576, 1024)  # Output perspective image size (height, width)
    out_h, out_w = out_shape

    focal_length_x = out_w / (2 * np.tan(np.radians(horizontal_fov) / 2))
    focal_length_y = out_h / (2 * np.tan(np.radians(vertical_fov) / 2))
    print(focal_length_x, focal_length_y)
    
    # Define camera intrinsic matrix
    camera_matrix = np.array([
        [focal_length_x, 0, out_w / 2],
        [0, focal_length_y, out_h / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    # Generate a random rotation matrix using random Euler angles
    def euler_to_rotmat(rx, ry, rz):
        """Converts Euler angles to a rotation matrix."""
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx

    # Random rotation angles in degrees
    # rx = np.radians(random.uniform(-30, 30))
    # ry = np.radians(random.uniform(-30, 30))
    # rz = np.radians(random.uniform(-30, 30))
    rx = -90 / 180 * 3.1415926
    ry = 0 / 180 * 3.1415926
    rz = 0
    rotation_matrix = euler_to_rotmat(rx, ry, rz).astype(np.float32)

    # Apply the transformation
    persp_img = wrap_panoramic_to_perspective(
        pano_img, camera_matrix, rotation_matrix, out_shape
    )

    # Save the result
    output_path = "/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/raw_vrs/ucl_campus/out_general_streetview/perspective_image/pano_wrap.jpg"
    cv2.imwrite(output_path, persp_img)
    print(f"Wrapped perspective image saved to: {output_path}")


if __name__ == "__main__":
    main()
