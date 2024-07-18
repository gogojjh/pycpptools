import numpy as np

class BaseNode:
	def __init__(self, id, trans=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0])):
		# Initialize the node with a given id and an empty list of edges
		self.id = id
		self.edges = []

		self.trans = trans
		self.quat = quat

		self.has_pose_gt = False
		self.trans_gt = np.zeros(3)
		self.quat_gt = np.array([0.0, 0.0, 0.0, 1.0])

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

	def add_edge(self, neighbor, weight):
		# Add an edge to the node by appending a tuple of the neighbor node and the weight
		self.edges.append((neighbor, weight))
