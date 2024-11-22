import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

def transform_point_cloud(points, transformation_matrix):
    """
    Apply a transformation to a point cloud.

    Parameters:
    points (numpy.ndarray): The point cloud as an (N, 3) array.
    transformation_matrix (numpy.ndarray): The 4x4 transformation matrix.

    Returns:
    numpy.ndarray: The transformed point cloud as an (N, 3) array.
    """
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    points_transformed = points_homogeneous @ transformation_matrix.T
    return points_transformed[:, :3]

def project_point_cloud(points, intrinsics, image_shape):
    """
    Project a point cloud onto an image plane.

    Parameters:
    points (numpy.ndarray): The point cloud as an (N, 3) array.
    intrinsics (numpy.ndarray): The camera intrinsic matrix.
    image_shape (tuple): The shape of the output image (height, width).

    Returns:
    numpy.ndarray: The projected depth image.
    """
    w, h = image_shape
    z = points[:, 2]
    x = (points[:, 0] * intrinsics[0, 0] / z + intrinsics[0, 2]).astype(np.int32)
    y = (points[:, 1] * intrinsics[1, 1] / z + intrinsics[1, 2]).astype(np.int32)

    depth_image = np.zeros((h, w))
    valid_mask = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (z > 0)
    depth_image[y[valid_mask], x[valid_mask]] = z[valid_mask]
    return depth_image	

def draw_images(image0, image1):
	# Create a figure and a set of subplots
	fig, axes = plt.subplots(1, 2, figsize=(12, 6))
	# Display the first image with a colorbar
	im1 = axes[0].imshow(image0, cmap='viridis')
	axes[0].axis('off')
	fig.colorbar(im1, ax=axes[0])
	# Display the second image with a colorbar
	im2 = axes[1].imshow(image1, cmap='viridis')
	axes[1].axis('off')
	fig.colorbar(im2, ax=axes[1])	
	plt.show()

def depth_alignment(A, B, delta=0.1):
    """
    Compute the scale factor s using the provided equation with a robust M-estimator, remove outliers
    Args:
                    A (np.ndarray): Reference matrix (depth_image1).
                    B (np.ndarray): Matrix to be scaled (depth_image2).
    Returns:
                    float: Computed scale factor.
    """

    def huber_loss(residual, delta):
        """
        Huber loss function.
        Args:
                        residual (np.ndarray): Residuals.
                        delta (float): Delta parameter for Huber loss.
        Returns:
                        float: Huber loss value.
        """
        return np.where(
            np.abs(residual) <= delta,
            0.5 * residual**2,
            delta * (np.abs(residual) - 0.5 * delta),
        )

    def objective_function(s):
        """
        Objective function to minimize.
        Args:
                        s (float): Scale factor.
        Returns:
                        float: Sum of Huber loss for residuals.
        """
        residual = A - s * B
        return np.sum(huber_loss(residual, delta))

    result = minimize(objective_function, x0=1.0)
    return result.x[0]

def compute_residual_matrix(A, B, s):
    def huber_loss(residual, delta=1.0):
        """
        Huber loss function.
        Args:
                        residual (np.ndarray): Residuals.
                        delta (float): Delta parameter for Huber loss.
        Returns:
                        float: Huber loss value.
        """
        return np.where(
            np.abs(residual) <= delta,
            0.5 * residual**2,
            delta * (np.abs(residual) - 0.5 * delta),
        )
    return huber_loss(A - s * B)

def plot_images(image1, image2, title1="Image 1", title2="Image 2", save_path=None):
    """
    Plot two images side by side with colorbars.

    Parameters:
    image1 (numpy.ndarray): The first image.
    image2 (numpy.ndarray): The second image.
    title1 (str): Title for the first image.
    title2 (str): Title for the second image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(image1, cmap="viridis")
    axes[0].set_title(title1)
    axes[0].axis("off")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(
        image2, cmap="viridis", vmin=im1.get_clim()[0], vmax=im1.get_clim()[1]
    )
    axes[1].set_title(title2)
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1])

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    from PIL import Image
    """Read the image and depth image"""
    rgb_image = np.array(Image.open('/Titan/dataset/data_apmp/hkustgz_campus/test/s00000/seq0/frame_00000.jpg')).astype(np.uint8)
    depth_image = np.array(Image.open('/Titan/dataset/data_apmp/hkustgz_campus/test/s00000/seq0/frame_00000.ray_neighbor.png')).astype(np.float32) / 1000.0

    # Provide the intrinsics matrix
    # fx, fy, cx, cy, image_width, image_height = 542.790830000, 542.790830000, 481.301150000, 271.850070000, 960, 540
    fx, fy, cx, cy, image_width, image_height = 913.896, 912.277, 638.954, 364.884, 1280, 720
    # fx, fy, cx, cy, image_width, image_height = 542.884160000, 542.884160000, 481.300450000, 271.850280000, 960, 540
    image_shape = (image_width, image_height)
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    points = depth_image_to_point_cloud(depth_image, intrinsics, image_shape)
    
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("/Titan/dataset/data_apmp/hkustgz_campus/test/s00000/seq0/frame_00000.ray_neighbor.pcd", pcd)