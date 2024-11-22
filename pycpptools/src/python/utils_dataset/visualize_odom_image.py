# extrinsics: frame_cam00 - base_link
# - Translation: [-0.507, 2.753, -1.736]
# - Rotation: in Quaternion [-0.585, 0.598, -0.370, -0.404]
#             in RPY (radian) [3.065, -1.159, -1.541]
#             in RPY (degree) [175.597, -66.401, -88.318]

# current_odom: in base_link

# K = [611.8597412109375, 0.0, 638.0593872070312, 0.0, 611.9406127929688, 367.66937255859375, 0.0, 0.0, 1.0
# height: 720, width: 1280

import rosbag
import cv2
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
import argparse

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from scipy.spatial.transform import Rotation

# Intrinsics
image_size = (720, 1280)  # Image size (height, width)
K = np.array([611.8597412109375, 0.0, 638.0593872070312, 0.0, 611.9406127929688, 367.66937255859375, 0.0, 0.0, 1.0]).reshape(3, 3)

# Extrinsics
trans_cam00_baselink = np.array([-0.507, 2.753, -1.736])
quat_cam00_baselink = np.array([-0.585, 0.598, -0.370, -0.404])

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

def convert_rosimg_to_cvimg(img_msg):
	bridge = CvBridge()
	cv_image = bridge.compressed_imgmsg_to_cv2(img_msg, "bgr8")
	return cv_image

def draw_trajectory_with_velocity(bag_file, path_output_image):
	idx = 0
	cnt = 0

	bag = rosbag.Bag(bag_file)
	for topic, msg, t in bag.read_messages(topics=['/rgb/image_rect_color/compressed']):
		cnt += 1
		if cnt % 1 != 0: continue

		image = convert_rosimg_to_cvimg(msg)
		time = msg.header.stamp.to_sec()
		T_cam00_baselink = convert_vec_to_matrix(trans_cam00_baselink, quat_cam00_baselink)	
		poses_filter = poses[(poses[:, 0] >= time - 0.1) & (poses[:, 0] <= time + 0.1)]
		T_w_baselink = convert_vec_to_matrix(poses_filter[0, 1:4], poses_filter[0, 4:8])
		T_cam00_ref_w = T_cam00_baselink @ np.linalg.inv(T_w_baselink)

		poses_filter = poses[(poses[:, 0] >= time - 5.0) & (poses[:, 0] <= time + 5.0)][::7]
		for pose_idx, pose in enumerate(poses_filter):
			trans = pose[1:4]
			quat = pose[4:8]
			vel = pose[8:11]
			T_w_baselink = convert_vec_to_matrix(trans, quat)
			T_cam00_ref_baselink = T_cam00_ref_w @ T_w_baselink
			if T_cam00_ref_baselink[2, 3] <= 0.0: continue

			x = (K @ T_cam00_ref_baselink[:3, 3].reshape(3, 1)).reshape(-1)
			x = x / x[2]
			if x[0] < 0 or x[0] >= image_size[1] or x[1] < 0 or x[1] >= image_size[0]:
				continue
			cv2.circle(image, (int(x[0]), int(x[1])), 8, (140, 3, 120), -1)
			if pose_idx % 1 == 0:
				cv2.putText(image, f'{vel[0]:.2f}m/s', (int(x[0]+10), int(x[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
		cv2.imwrite(f'{path_output_image}/output_{idx:06d}.png', image)
		print(f'Saved {path_output_image}/output_{idx:06d}.png')
		idx += 1

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Draw trajectory with velocity markers')
	parser.add_argument('--bag_file', type=str, help='Path to the ROS bag file')
	parser.add_argument('--path_input_odom', type=str, help='Path to the input odometry file')
	parser.add_argument('--path_output_image', type=str, help='Path to the output image file')
	args = parser.parse_args()

	bag_file = args.bag_file
	poses = np.loadtxt(args.path_input_odom)

	# Draw the trajectory with velocity markers and save it to an image
	draw_trajectory_with_velocity(bag_file, args.path_output_image)
