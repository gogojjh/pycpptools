import numpy as np
import matplotlib.pyplot as plt

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
