#! /usr/bin/env python3

# Import necessary libraries
import gtsam
import numpy as np
from scipy.spatial.transform import Rotation

def convert_vec_gtsam_pose3(trans, quat, mode='xyzw'):
    from scipy.spatial.transform import Rotation
    if mode=='xyzw':
        pose3 = gtsam.Pose3(gtsam.Rot3(quat[3], quat[0], quat[1], quat[2]), trans)
    else:
        pose3 = gtsam.Pose3(gtsam.Rot3(quat[0], quat[1], quat[2], quat[3]), trans)
    return pose3

# Function to convert position and quaternion vectors to a transformation matrix
# vec_p: position vector (x, y, z)
# vec_q: quaternion vector (qw, qx, qy, qz)
def convert_vec_to_matrix(vec_p, vec_q, mode='xyzw'):
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
def convert_matrix_to_vec(tf_matrix, mode='xyzw'):
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

def add_gaussian_noise_to_pose(trans, quat, mean=0, stddev=0.1, mode='xyzw'):
	if mode == 'xyzw':
		noisy_trans = np.array([x + np.random.normal(mean, stddev) for x in trans])
		rot = Rotation.from_quat(quat)
		euler = rot.as_euler('xyz', degrees=False)
		noisy_euler = [x + np.random.normal(mean, stddev) for x in euler]
		noisy_rot = Rotation.from_euler('xyz', noisy_euler, degrees=False)
		noisy_quat = noisy_rot.as_quat()
	elif mode == 'wxyz':
		noisy_trans = np.array([x + np.random.normal(mean, stddev) for x in trans])
		rot = Rotation.from_quat(np.roll(quat, -1))
		euler = rot.as_euler('xyz', degrees=False)
		noisy_euler = [x + np.random.normal(mean, stddev) for x in euler]
		noisy_rot = Rotation.from_euler('xyz', noisy_euler, degrees=False)
		noisy_quat = np.roll(noisy_rot.as_quat(), 1)   
	return noisy_trans, noisy_quat

def compute_relative_dis(last_t, last_quat, curr_t, curr_quat, mode='xyzw'):
	if mode == 'xyzw':
		rot1 = Rotation.from_quat(last_quat)
		rot2 = Rotation.from_quat(curr_quat)
		rel_rot = rot2 * rot1.inv()
		dis_angle = np.linalg.norm(rel_rot.as_euler('xyz', degrees=True))
		dis_trans = np.linalg.norm(rot1.inv().apply(last_t - curr_t))
	if mode == 'wxyz':
		rot1 = Rotation.from_quat(np.roll(last_quat, -1))
		rot2 = Rotation.from_quat(np.roll(curr_quat, -1))   
		rel_rot = rot2 * rot1.inv()
		dis_angle = np.linalg.norm(rel_rot.as_euler('xyz', degrees=True))
		dis_trans = np.linalg.norm(rot1.inv().apply(last_t - curr_t))
	return dis_trans, dis_angle

def compute_relative_dis_TF(last_T, curr_T):
	rel_T = np.linalg.inv(last_T) @ curr_T
	rel_rot = Rotation.from_matrix(rel_T[:3, :3])
	dis_angle = np.linalg.norm(rel_rot.as_euler('xyz', degrees=True))
	dis_trans = np.linalg.norm(rel_T[:3, 3])
	return dis_trans, dis_angle

def test_tools_eigen():
	# Define a position vector
	vec_p = np.array([0.1, 0.3, 0.6])
	# Define a quaternion vector representing a rotation
	vec_q = np.array([0.0, 0.0, 0.0, 1.0])
	# Convert the vectors to a transformation matrix
	tf = convert_vec_to_matrix(vec_p, vec_q, mode='xyzw')
	# Print the resulting transformation matrix
	print(tf)

	# Convert the transformation matrix back to position and quaternion vectors
	vec_p, vec_q = convert_matrix_to_vec(tf, mode='xyzw')
	# Print the resulting position vector
	print(vec_p)
	# Print the resulting quaternion vector
	print(vec_q)

	# Do transformation
	vec_p1 = np.array([0.037981, 0.065102, -0.114576])
	vec_q1 = np.array([-0.000782, -0.131110, -0.009875, 0.991318])
	T1 = convert_vec_to_matrix(vec_p1, vec_q1, mode='wxyz')

	vec_p2 = np.array([0.011, -0.060, -0.015])
	vec_q2 = np.array([0.000, -0.025, 0.000, 1.000])
	T2 = convert_vec_to_matrix(vec_p2, vec_q2, mode='xyzw')

	T = T1 @ T2
	vec_p, vec_q = convert_matrix_to_vec(T, mode='wxyz')
	out_str = 'quat_opt: ' + ' '.join([f'{x:05f}' for x in vec_q])
	out_str += ', trans_opt: ' + ' '.join([f'{x:05f}' for x in vec_p])
	print(out_str)

# Main function to test the conversion functions
if __name__ == '__main__':
	test_tools_eigen()
