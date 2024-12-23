import os
import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def load_image(filepath):
    """Load the panoramic image."""
    return cv2.imread(filepath)

def resize_image(image, scale):
    """Resize the image by a given scale factor."""
    height, width = image.shape[:2]
    new_width, new_height = int(width * scale), int(height * scale)
    return cv2.resize(image, (new_width, new_height)), new_width, new_height

def extract_features(image):
    """Extract ORB features from the image."""
    orb = cv2.ORB_create(nfeatures=4096)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def visualize_features(image, keypoints, output_path):
    """
    Visualize ORB features on the panoramic image.
    
    Args:
        image (np.ndarray): The panoramic image.
        keypoints (list): List of cv2.KeyPoint objects representing detected features.
    """
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)
    cv2.imwrite(os.path.join(output_path, 'pano_orb.jpg'), image_with_keypoints)

def pixel_to_spherical(feature_points, width, height):
    """Map pixel coordinates to spherical angles."""
    phi = 2 * np.pi * feature_points[:, 0] / width - np.pi
    theta = np.pi * (feature_points[:, 1] / height - 0.5)
    return phi, theta

def spherical_to_cartesian(phi, theta):
    """Convert spherical coordinates to 3D Cartesian coordinates."""
    X = np.cos(theta) * np.cos(phi)
    Y = np.cos(theta) * np.sin(phi)
    Z = np.sin(theta)
    return np.stack((X, Y, Z), axis=1)

def cartesian_to_spherical(x, y, z):
    """Convert Cartesian coordinates to spherical coordinates."""
    phi = np.arctan2(y, x)
    theta = np.arcsin(z)
    return phi, theta

def cluster_features(points_3D, num_clusters):
    """Cluster 3D points on the sphere using K-means clustering with additional distance constraints."""
    max_attempts = 10
    best_centers = None
    best_score = 0

    for _ in range(max_attempts):
        clustering = KMeans(n_clusters=num_clusters, random_state=None).fit(points_3D)
        cluster_centers = clustering.cluster_centers_
        
        # Calculate pairwise distances between cluster centers
        distances = cdist(cluster_centers, cluster_centers)
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances

        # Score based on minimum pairwise distance (maximize separation)
        min_distance = np.min(distances)
        if min_distance > best_score:
            best_score = min_distance
            best_centers = cluster_centers

    return best_centers

def generate_view_parameters(cluster_centers):
    """Generate perspective view parameters based on cluster centers."""
    cluster_phi, cluster_theta = cartesian_to_spherical(cluster_centers[:, 0],
                                                        cluster_centers[:, 1],
                                                        cluster_centers[:, 2])
    perspective_views = []
    for phi_c, theta_c in zip(cluster_phi, cluster_theta):
        view_params = { # degree
            # "heading": np.degrees(phi_c),
            # "pitch": np.degrees(theta_c),
            "heading": 0.0,
            "pitch": -90.0,
        }
        perspective_views.append(view_params)
    print(perspective_views)
    return perspective_views

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

def process_panorama(filepath, output_path, num_clusters, output_size, fov_h, fov_v, resize_scale=0.5):
    """Main function to process a panoramic image and generate perspective views."""
    # Step 1: Load image
    image = load_image(filepath)
    print(f"Raw image size: {image.shape[0]}x{image.shape[1]}")

    # Step 2: Resize image for clustering
    resized_image, new_width, new_height = resize_image(image, resize_scale)
    print(f"Resize image size: {resized_image.shape[0]}x{resized_image.shape[1]}")

    # Step 3: Extract features
    keypoints, _ = extract_features(resized_image)
    print(f"Extract {len(keypoints)} ORB Features")
    visualize_features(resized_image, keypoints, output_path)

    # Step 4: Convert to spherical coordinates
    feature_points = np.array([kp.pt for kp in keypoints])  # (u, v) coordinates  
    phi, theta = pixel_to_spherical(feature_points, new_width, new_height)
    points_3D = spherical_to_cartesian(phi, theta)

    # Step 5: Cluster features
    cluster_centers = cluster_features(points_3D, num_clusters)

    # Step 6: Generate perspective view parameters
    perspective_views = generate_view_parameters(cluster_centers)

    # Step 7: Generate perspective views
    perspective_images = []
    perspective_matrices = []
    output_height, output_width = output_size[0], output_size[1]
    for view in perspective_views:
        # Calculate camera and rotation matrices
        K = np.array([[output_width / (2 * np.tan(np.radians(fov_h / 2))), 0, output_width / 2],
                      [0, output_height / (2 * np.tan(np.radians(fov_v / 2))), output_height / 2],
                      [0, 0, 1]])
        R = cv2.Rodrigues(np.array([np.radians(view["pitch"]), 0, np.radians(-view["heading"])]))[0]
        
        # Generate perspective view
        perspective_img = wrap_panoramic_to_perspective(image, K, R, out_shape=output_size)
        perspective_images.append(perspective_img)
        perspective_matrices.append((R, K))

    return perspective_images, perspective_matrices

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate perspective views from a panoramic image.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the equirectangular panoramic image.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the perspective image.")
    parser.add_argument("--num_clusters", type=int, required=True, help="Number of perspective views to generate.")
    parser.add_argument("--output_size", type=int, nargs=2, required=True, help="Output resolution of perspective images (height, width).")
    parser.add_argument("--hfov", type=float, default=120.0, required=True, help="Horizontal field of view in degrees.")
    parser.add_argument("--vfov", type=float, default=45.0, required=True, help="Vertical field of view in degrees.")
    args = parser.parse_args()

    perspective_views, matrices = process_panorama(
        args.input_path,
        args.output_path,
        args.num_clusters,
        tuple(args.output_size),
        args.hfov,
        args.vfov
    )

    # Save the perspective views
    for i, img in enumerate(perspective_views):
        cv2.imwrite(f"{args.output_path}/perspective_view_{i}.jpg", img)
        print(f"View {i}:\nR=\n{matrices[i][0]}\nK=\n{matrices[i][1]}")
