import numpy as np
import pycpptools.src.python.utils_math as pytool_math

class BaseNode:
	def __init__(self, id, trans=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0])):
		# Initialize the node with a given id and an empty list of edges
		self.id = id
		self.edges = []    # [(nodeB, weight), ...]

		self.trans = trans # xyzw
		self.quat = quat   # xyzw

		self.has_pose_gt = False
		self.trans_gt = np.zeros(3)
		self.quat_gt = np.array([0.0, 0.0, 0.0, 1.0])

		# Next node using in the shortest path
		self.next_node = None

	def __str__(self):
		# Return a string representation of the node, including its id and number of edges
		out_str = f'Node ID: {self.id} with edge number: {len(self.edges)}'
		return out_str

	def __lt__(self, other):
		# Define the less than operator for comparing nodes based on their id
		return self.id < other.id

	def set_pose(self, trans, quat):
		self.trans = trans
		self.quat = quat

	def set_pose_gt(self, trans_gt, quat_gt):
		self.has_pose_gt = True
		self.trans_gt = trans_gt
		self.quat_gt = quat_gt

	def add_edge(self, next_node, weight):
		# Add an edge to the node by appending a tuple of the next_node and the weight
		self.edges.append((next_node, weight))

	def add_next_node(self, next_node):
		self.next_node = next_node

	def get_next_node(self):
		return self.next_node
	
	def compute_distance(self, node):
		# Compute the Euclidean distance between two nodes based on their poses
		dis_trans, dis_angle = \
			pytool_math.tools_eigen.compute_relative_dis(self.trans, self.quat, node.trans, node.quat)
		return dis_trans, dis_angle
		