#!/usr/bin/python3

import argparse

import rospy
import rosbag
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from utils_ros.tools_ros_msg_conversion import convert_vec_to_rosodom, convert_vec_to_rospose, convert_vec_to_ros_tfmsg
from utils_math.tools_eigen import convert_matrix_to_vec, convert_vec_to_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--input_odometry_file', type=str, help='/tmp/odom.txt')
parser.add_argument('--output_bag_path', type=str, help='/tmp/output_bag_path.bag')
parser.add_argument('--format', type=str, help='TUM, KITTI', default='TUM')
args = parser.parse_args()
print("Arguments:\n{}".format('\n'.join(
		['-{}: {}'.format(k, v) for k, v in args.__dict__.items()])))

def save_odom_to_rosbag(args):
    outbag = rosbag.Bag(args.output_bag_path, 'w')

    poses = np.loadtxt(args.input_odometry_file)
    path = Path()
    for pose in poses:
        timestamp = pose[0]
        trans = pose[1:4]
        quat = pose[4:]

        header = Header()
        header.frame_id = 'map'
        header.stamp = rospy.Time.from_sec(timestamp)

        odom = convert_vec_to_rosodom(trans, quat, header, 'zed2_left_camera_optical_frame')
        outbag.write('/AirSLAM/odometry', odom, header.stamp)    

        pose_msg = convert_vec_to_rospose(trans, quat, header)
        path.header = header
        path.poses.append(pose_msg)
        outbag.write('/AirSLAM/path', path, header.stamp)

        # tf_msg = convert_vec_to_ros_tfmsg(trans, quat, header, 'zed2_left_camera_optical_frame')
        # outbag.write('/tf', tf_msg, header.stamp)
    outbag.close()

if __name__ == '__main__':
	rospy.init_node('tools_bag_save_odom', anonymous=True)
	if args.format == 'TUM':
		save_odom_to_rosbag(args)
	else:
		print('Unsupported format: {}'.format(args.format))
		exit(1)