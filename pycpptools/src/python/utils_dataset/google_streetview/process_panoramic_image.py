import argparse
import cv2
import numpy as np
import os

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

    # NOTE(gogojjh):
    uf = (theta + np.pi) / (2 * np.pi) * equirect_width
    vf = (np.pi/2 + phi) / np.pi * equirect_height

    # Sample the pixels
    perspective = cv2.remap(img, uf.astype(np.float32), vf.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return perspective

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

    # Construct the camera intrinsic matrix
    camera_matrix = construct_camera_matrix(args.hfov, args.vfov, args.output_size)
    print("Camera Intrinsic Matrix:")
    print(camera_matrix)

    # Load each panoramic image
    list_filenames = os.listdir(args.input_path)
    list_filenames.sort()
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
        for idx, R in enumerate(rotation_matrices):
            perspective = equirectangular_to_perspective(img_resized, camera_matrix, R, args.output_size)
            output_filename = os.path.join(args.output_path, f"perspective_view_{img_ind}_{idx}.png")
            cv2.imwrite(output_filename, perspective)
            # print(f"Saved perspective image {idx+1} to {output_filename}")
        # break

if __name__ == "__main__":
    main()
