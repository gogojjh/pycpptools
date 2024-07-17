#! /usr/bin/env python3

import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def read_image(file_path):
    """
    Read an image and convert it to a numpy array.
    
    Parameters:
    file_path (str): The path to the image file.
    
    Returns:
    numpy.ndarray: The image as a numpy array.
    """
    image = Image.open(file_path)
    return np.array(image)

def plot_images(image1, image2, title1="Image 1", title2="Image 2"):
    """
    Plot two images side by side with colorbars.
    
    Parameters:
    image1 (numpy.ndarray): The first image.
    image2 (numpy.ndarray): The second image.
    title1 (str): Title for the first image.
    title2 (str): Title for the second image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axes[0].imshow(image1, cmap='viridis')
    axes[0].set_title(title1)
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(image2, cmap='viridis', vmin=im1.get_clim()[0], vmax=im1.get_clim()[1])
    axes[1].set_title(title2)
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1])

    plt.show()

def compute_scale_factor(A, B):
    """
    Compute the scale factor s using the provided equation with a robust M-estimator, remove outliers
    
    Args:
        A (np.ndarray): Reference matrix (depth_image1).
        B (np.ndarray): Matrix to be scaled (depth_image2).
    
    Returns:
        float: Computed scale factor.
    """
    def huber_loss(residual, delta=1.0):
        """
        Huber loss function.
        
        Args:
            residual (np.ndarray): Residuals.
            delta (float): Delta parameter for Huber loss.
        
        Returns:
            float: Huber loss value.
        """
        return np.where(np.abs(residual) <= delta, 0.5 * residual**2, delta * (np.abs(residual) - 0.5 * delta))

    def objective_function(s):
        """
        Objective function to minimize.
        
        Args:
            s (float): Scale factor.
        
        Returns:
            float: Sum of Huber loss for residuals.
        """
        residual = A - s * B
        return np.sum(huber_loss(residual))
    
    result = minimize(objective_function, x0=1.0)
    return result.x[0]

def main():
    parser = argparse.ArgumentParser(description="Process depth images.")
    parser.add_argument('--ref_depth_image', type=str, help="Path to the first depth image (reference).")
    parser.add_argument('--target_depth_image', type=str, help="Path to the second depth image to be scaled.")
    args = parser.parse_args()
    
    # Read the images
    A = read_image(args.ref_depth_image)
    B = read_image(args.target_depth_image)
    print(A.shape)

    # Set zero value of pixels that is outside the range ([0.05m - 5.5m])
    min_th, max_th = 50, 5500
    A_filter = np.zeros_like(A)
    B_filter = np.zeros_like(B)

    mask = (A > min_th) & (A < max_th)
    A_filter[mask] = A[mask]
    B_filter[mask] = B[mask]
    
    # Compute the scaling factor
    s = compute_scale_factor(A_filter, B_filter)
    print(f'Scale Factor: {s:.3f}')

    total_dis_before_scaling = np.linalg.norm(A_filter - B_filter, 'fro')
    mean_dis_before_scaling = total_dis_before_scaling / np.size(A_filter)
    total_dis_after_scaling = np.linalg.norm((A_filter - s * B_filter), 'fro')
    mean_dis_after_scaling = total_dis_after_scaling / np.size(A_filter)
    print(f'Total Disp before Scaling: {total_dis_before_scaling:.3f}, ', 
          f'Mean Disp before Scaling: {mean_dis_before_scaling:.3f}')
    print(f'Total Disp after Scaling: {total_dis_after_scaling:.3f}, ', 
          f'Mean Disp after Scaling: {mean_dis_after_scaling:.3f}')
    print(f'Reduce Ratio: {mean_dis_before_scaling / mean_dis_after_scaling:.3f}')
    
    # Scale the second image
    B_scaled = B_filter * s
    
    # Plot
    plot_images(A_filter, B_filter, title1="Depth Image 1 (Reference)", title2="Depth Image 2 (Original)")
    plot_images(A_filter, B_filter * s, title1="Depth Image 1 (Reference)", title2="Depth Image 2 (Scaled)")
   
    plot_images(A_filter, np.abs(A_filter - B_filter), title1="Depth Image 1 (Reference)", title2="Error Map")
    plot_images(A_filter, np.abs(A_filter - B_scaled), title1="Depth Image 1 (Reference)", title2="Error Map")

if __name__ == "__main__":
    main()
