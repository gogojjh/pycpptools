#!/usr/bin/python3

import argparse
import os

import rospy
import rosbag
from nav_msgs.msg import Odometry
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--in_bag_path', type=str, help='/tmp/inbag_path.bag')
parser.add_argument('--out_pose_path', type=str, help='/tmp/pose.txt')
parser.add_argument('--topic_odom', type=str, default=None, help='/current_odom')
parser.add_argument('--topic_path', type=str, default=None, help='/current_path')
parser.add_argument('--format', type=str, default='TUM', help='TUM, KITTI')
args = parser.parse_args()
print("Arguments:\n{}".format('\n'.join(
		['-{}: {}'.format(k, v) for k, v in args.__dict__.items()])))

def save_path_tum_format(args):
	bag = rosbag.Bag(args.in_bag_path, 'r')

	end_time = None
	for topic, msg, t in bag.read_messages():
		end_time = t
	print(end_time.secs + end_time.nsecs / 1e9)

	for topic, msg, t in bag.read_messages(topics=[args.topic_path]):
		print(msg.header.stamp.secs + msg.header.stamp.nsecs / 1e9, 
			 (msg.header.stamp.secs + msg.header.stamp.nsecs / 1e9 - end_time.secs - end_time.nsecs / 1e9))
		if abs(msg.header.stamp.secs + msg.header.stamp.nsecs / 1e9 - end_time.secs - end_time.nsecs / 1e9) > 3.0: 
			continue
		pose_list = []
		for pose_msg in msg.poses:
			timestamp = pose_msg.header.stamp.secs + pose_msg.header.stamp.nsecs / 1e9
			tx = pose_msg.pose.position.x
			ty = pose_msg.pose.position.y
			tz = pose_msg.pose.position.z
			qx = pose_msg.pose.orientation.x
			qy = pose_msg.pose.orientation.y
			qz = pose_msg.pose.orientation.z
			qw = pose_msg.pose.orientation.w
			pose_list.append([timestamp, tx, ty, tz, qx, qy, qz, qw])
		break
	bag.close()
	np.savetxt(args.out_pose_path, np.array(pose_list), '%.9f')

def save_odom_tum_format(args):
	bag = rosbag.Bag(args.in_bag_path, 'r')
	pose_list = []
	for topic, msg, t in bag.read_messages(topics=[args.topic_odom]):
		timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs / 1e9
		tx = msg.pose.pose.position.x
		ty = msg.pose.pose.position.y
		tz = msg.pose.pose.position.z
		qx = msg.pose.pose.orientation.x
		qy = msg.pose.pose.orientation.y
		qz = msg.pose.pose.orientation.z
		qw = msg.pose.pose.orientation.w
		pose_list.append([timestamp, tx, ty, tz, qx, qy, qz, qw])
	bag.close()
	np.savetxt(args.out_pose_path, np.array(pose_list), '%.9f')

if __name__ == '__main__':
	rospy.init_node('tools_bag_save_odom', anonymous=True)
	if args.format == 'TUM':
		if args.topic_odom is not None:
			save_odom_tum_format(args)
		elif args.topic_path is not None:
			save_path_tum_format(args)
	else:
		print('Unsupported format: {}'.format(args.format))
		exit(1)