#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from nav_msgs.msg import Odometry, Path
import tf2_ros
import tf.transformations as tf
import numpy as np

from tools_ros_msg_conversion import *
from utils_algorithm.base_node import BaseNode as Node
from utils_algorithm.base_graph import BaseGraph as Graph

def create_node_marker(node):
  marker = Marker()
  marker.header.frame_id = "map"
  marker.header.stamp = rospy.Time.now()
  marker.ns = "nodes"
  marker.id = node.id
  marker.type = Marker.CUBE
  marker.action = Marker.ADD
  marker.pose.position.x = node.trans_w_node[0]
  marker.pose.position.y = node.trans_w_node[1]
  marker.pose.position.z = node.trans_w_node[2]
  marker.pose.orientation.x = 0.0
  marker.pose.orientation.y = 0.0
  marker.pose.orientation.z = 0.0
  marker.pose.orientation.w = 1.0
  marker.scale.x = 0.5
  marker.scale.y = 0.5
  marker.scale.z = 0.5
  marker.color.a = 1.0
  marker.color.r = 0.0
  marker.color.g = 1.0
  marker.color.b = 0.0
  return marker

def create_text_marker(text_id, position, text):
  marker = Marker()
  marker.header.frame_id = "map"
  marker.header.stamp = rospy.Time.now()
  marker.ns = "text"
  marker.id = text_id
  marker.type = Marker.TEXT_VIEW_FACING
  marker.action = Marker.ADD
  marker.pose.position.x = position[0]
  marker.pose.position.y = position[1]
  marker.pose.position.z = position[2] + 0.5
  marker.scale.z = 0.5
  marker.color.a = 1.0
  marker.color.r = 1.0
  marker.color.g = 1.0
  marker.color.b = 1.0
  marker.text = text
  return marker

def create_edge_marker(node1, node2, edge_id, weight):
  marker = Marker()
  marker.header.frame_id = "map"
  marker.header.stamp = rospy.Time.now()
  marker.ns = "edges"
  marker.id = edge_id
  marker.type = Marker.LINE_STRIP
  marker.action = Marker.ADD
  marker.scale.x = weight / 20.0
  marker.color.a = 0.7
  marker.color.r = 1.0
  marker.color.g = 0.0
  marker.color.b = 0.0

  start_point = Point()
  start_point.x = node1.trans_w_node[0]
  start_point.y = node1.trans_w_node[1]
  start_point.z = node1.trans_w_node[2]

  end_point = Point()
  end_point.x = node2.trans_w_node[0]
  end_point.y = node2.trans_w_node[1]
  end_point.z = node2.trans_w_node[2]

  marker.points.append(start_point)
  marker.points.append(end_point)
  return marker

def publish_graph(graph, pub_graph):
  marker_array = MarkerArray()
  for node_id, node in graph.nodes.items():
    node_marker = create_node_marker(node)
    marker_array.markers.append(node_marker)

    text_marker = create_text_marker(node.id, node.trans_w_node, f'{node_id}')
    marker_array.markers.append(text_marker)

  edge_id = 0
  for _, node in graph.nodes.items():
    for (next_node, weight) in node.edges:
      edge_marker = create_edge_marker(node, next_node, edge_id, weight)
      marker_array.markers.append(edge_marker)
      edge_id += 1

  pub_graph.publish(marker_array)



class TestRosVisualization:
  def __init__(self):
    pass

  def run_test(self):
    rospy.init_node('test_ros_visualization', anonymous=True)

    pub_graph = rospy.Publisher('/topo_graph', MarkerArray, queue_size=10)
    pub_odom = rospy.Publisher('/odom', Odometry, queue_size=10)
    pub_path = rospy.Publisher('/path', Path, queue_size=10)
    br = tf2_ros.TransformBroadcaster()    

    graph = Graph()
    graph.add_node(Node(0, np.random.rand(3, 1) * 5.0))
    graph.add_node(Node(1, np.random.rand(3, 1) * 5.0))
    graph.add_node(Node(2, np.random.rand(3, 1) * 5.0))
    graph.add_node(Node(3, np.random.rand(3, 1) * 5.0))
    graph.add_node(Node(4, np.random.rand(3, 1) * 5.0))
    graph.add_edge(graph.get_node(0), graph.get_node(1), 1)
    graph.add_edge(graph.get_node(0), graph.get_node(2), 2)
    graph.add_edge(graph.get_node(0), graph.get_node(3), 3)
    graph.add_edge(graph.get_node(3), graph.get_node(4), 5)
        
    trans = np.random.rand(3, 1) * 5.0
    quat = np.random.rand(4, 1)
    quat /= np.linalg.norm(quat)

    path_msg = Path()

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
      # Publish graph
      publish_graph(graph, pub_graph)

      # Publish odometry, path and tf messages
      header = Header()
      header.stamp = rospy.Time.now()
      header.frame_id = "map"
      child_frame_id = "camera"

      odom_msg = convert_vec_to_rosodom(trans, quat, header, child_frame_id)
      pub_odom.publish(odom_msg)
      
      pose_msg = convert_vec_to_rospose(trans, quat, header)
      path_msg.header = header
      path_msg.poses.append(pose_msg)
      pub_path.publish(path_msg)

      tf_msg = convert_vec_to_rostf(trans, quat, header, child_frame_id)
      br.sendTransform(tf_msg)

      # Sleep to control the rate
      rate.sleep()

if __name__ == '__main__':
  test_ros_visualization = TestRosVisualization()
  test_ros_visualization.run_test()
