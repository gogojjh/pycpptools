import numpy as np

class BaseNode:
	def __init__(self, id, trans_w_node=np.zeros(3), quat_w_node=np.array([0.0, 0.0, 0.0, 1.0])):
		# Initialize the node with a given id and an empty list of edges
		self.id = id
		self.edges = []

		self.trans_w_node = trans_w_node
		self.quat_w_node = quat_w_node

	def __str__(self):
		# Return a string representation of the node, including its id and number of edges
		out_str = f'Node ID: {self.id} with edge number: {len(self.edges)}'
		return out_str

	def __lt__(self, other):
		# Define the less than operator for comparing nodes based on their id
		return self.id < other.id

	def set_pose(self, trans_w_node, quat_w_node):
		self.trans_w_node = trans_w_node
		self.quat_w_node = quat_w_node

	def add_edge(self, neighbor, weight):
		# Add an edge to the node by appending a tuple of the neighbor node and the weight
		self.edges.append((neighbor, weight))
