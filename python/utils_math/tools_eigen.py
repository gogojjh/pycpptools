#! /usr/bin/env python3

# Import necessary libraries
import numpy as np
from scipy.spatial.transform import Rotation

# Function to convert position and quaternion vectors to a transformation matrix
# vec_p: position vector (x, y, z)
# vec_q: quaternion vector (qw, qx, qy, qz)
def convert_vec_to_matrix(vec_p, vec_q, mode='wxyz'):
    # Initialize a 4x4 identity matrix
    tf = np.eye(4)
    if mode == 'xyzw':
        # Set the rotation part of the transformation matrix using the quaternion
        tf[:3, :3] = Rotation.from_quat(vec_q).as_matrix()
        # Set the translation part of the transformation matrix
        tf[:3, 3] = vec_p
    elif mode == 'wxyz':
        # Set the rotation part of the transformation matrix using the quaternion
        tf[:3, :3] = Rotation.from_quat(np.roll(vec_q, -1)).as_matrix()
        # Set the translation part of the transformation matrix
        tf[:3, 3] = vec_p
    return tf

# Function to convert a transformation matrix back to position and quaternion vectors
# tf_matrix: 4x4 transformation matrix
# vec_p: position vector (x, y, z)
# vec_q: quaternion vector (qw, qx, qy, qz)
def convert_matrix_to_vec(tf_matrix, mode='wxyz'):
    if mode == 'xyzw':
        # Extract the translation vector from the matrix
        vec_p = tf_matrix[:3, 3]
        # Extract the rotation part of the matrix and convert it to a quaternion
        vec_q = Rotation.from_matrix(tf_matrix[:3, :3]).as_quat()
    if mode == 'wxyz':
        # Extract the translation vector from the matrix
        vec_p = tf_matrix[:3, 3]
        # Extract the rotation part of the matrix and convert it to a quaternion
        vec_q = np.roll(Rotation.from_matrix(tf_matrix[:3, :3]).as_quat(), 1)
    return vec_p, vec_q

def add_gaussian_noise_to_pose(trans, quat, mean=0, stddev=0.1):
    noisy_trans = np.array([x + np.random.normal(mean, stddev) for x in trans])

    rot = Rotation.from_quat(np.roll(quat, -1))
    euler = rot.as_euler('xyz', degrees=False)
    noisy_euler = [x + np.random.normal(mean, stddev) for x in euler]
    noisy_rot = Rotation.from_euler('xyz', noisy_euler, degrees=False)
    noisy_quat = np.roll(noisy_rot.as_quat(), 1)
    
    return noisy_trans, noisy_quat

def compute_relative_dis(last_t, last_quat, curr_t, curr_quat):
    rot1 = Rotation.from_quat(np.roll(last_quat, -1)) # [qw qx qy qz] -> [qx qy qz qw]
    rot2 = Rotation.from_quat(np.roll(curr_quat, -1))   
    rel_rot = rot2 * rot1.inv()
    dis_angle = np.linalg.norm(rel_rot.as_euler('xyz', degrees=True))
    dis_trans = np.linalg.norm(rot1.inv().apply(last_t - curr_t))
    return dis_trans, dis_angle

def test_tools_eigen():
    # Define a position vector
    vec_p = np.array([0.1, 0.3, 0.6])
    # Define a quaternion vector representing a rotation
    vec_q = np.array([0.0, 0.0, 0.70710678, 0.70710678])
    # Convert the vectors to a transformation matrix
    tf = convert_vec_to_matrix(vec_p, vec_q)
    # Print the resulting transformation matrix
    print(tf)

    # Convert the transformation matrix back to position and quaternion vectors
    vec_p, vec_q = convert_matrix_to_vec(tf)
    # Print the resulting position vector
    print(vec_p)
    # Print the resulting quaternion vector
    print(vec_q)

# Main function to test the conversion functions
if __name__ == '__main__':
    test_tools_eigen()
