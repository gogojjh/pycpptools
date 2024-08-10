#!/usr/bin/python3

import argparse
import os

import rospy
import rosbag
from nav_msgs.msg import Odometry
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_bag_path', type=str, help='/tmp/inbag_path.bag')
parser.add_argument('--output_odom_path', type=str, help='/tmp/odom.txt')
parser.add_argument('--topic_odometry', type=str, help='/current_odom')
parser.add_argument('--format', type=str, help='TUM, KITTI')
args = parser.parse_args()
print("Arguments:\n{}".format('\n'.join(
		['-{}: {}'.format(k, v) for k, v in args.__dict__.items()])))

def save_odom_tum_format(args):
	bag = rosbag.Bag(args.input_bag_path, 'r')
	odom_list = []
	for topic, msg, t in bag.read_messages(topics=[args.topic_odometry]):
		timestamp = msg.header.stamp.secs + msg.header.stamp.nsecs / 1e9
		tx = msg.pose.pose.position.x
		ty = msg.pose.pose.position.y
		tz = msg.pose.pose.position.z
		qx = msg.pose.pose.orientation.x
		qy = msg.pose.pose.orientation.y
		qz = msg.pose.pose.orientation.z
		qw = msg.pose.pose.orientation.w
		odom_list.append([timestamp, tx, ty, tz, qx, qy, qz, qw])
	bag.close()
	np.savetxt(args.output_odom_path, np.array(odom_list), '%.9f')

if __name__ == '__main__':
	rospy.init_node('tools_bag_save_odom', anonymous=True)
	if args.format == 'TUM':
		save_odom_tum_format(args)
	else:
		print('Unsupported format: {}'.format(args.format))
		exit(1)