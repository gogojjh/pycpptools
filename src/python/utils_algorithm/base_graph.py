import numpy as np

class BaseGraph:
  # Initialize an empty dictionary to store nodes
  def __init__(self):
    self.nodes = {}

  def __str__(self):
    num_edge = 0
    for node_id, node in self.nodes.items():
      num_edge += len(node.edges)
    out_str = f"Graph has {len(self.nodes)} nodes with {num_edge} edges"
    return out_str

  def read_edge_list(self, path_edge_list):
    edges_A_B_weight = np.loadtxt(path_edge_list, dtype=float)
    for edge in edges_A_B_weight:
      if (self.get_node(edge[0]) is not None) and (self.get_node(edge[1]) is not None):
        node1 = self.get_node(edge[0])
        node2 = self.get_node(edge[1])
        self.add_edge_directed(node1, node2, edge[2])
        self.add_edge_directed(node2, node1, edge[2])

  # Add a new node to the graph if it doesn't already exist
  def add_node(self, new_node):
    if not self.contain_node(new_node):
      self.nodes[new_node.id] = new_node

  def add_edge_undirected(self, from_node, to_node, weight):
    # Add an edge between two nodes if both nodes exist in the graph
    if self.contain_node(from_node) and self.contain_node(to_node):
      from_node.add_edge(to_node, weight)
      to_node.add_edge(from_node, weight)  # Assuming undirected graph

  # Add an edge between two nodes if both nodes exist in the graph
  def add_edge_directed(self, from_node, to_node, weight):
    if self.contain_node(from_node) and self.contain_node(to_node):
      for edge in from_node.edges:
        # Edge already exists with the same weight, do not add again
        if (edge[0].id == to_node.id) and (edge[1] == weight):
          return
        # Replace the current lighter edge with the new edge
        if (edge[0].id == to_node.id) and (edge[1] > weight):
          from_node.edges.remove(edge)  # Remove the current heavier edge
          break
      from_node.add_edge(to_node, weight)  # Add the new edge with the specified weight

  # Return the node with the given id if it exists, otherwise return None
  def get_node(self, id):
    if id in self.nodes:
      return self.nodes[id]
    else:
      return None

  # Return the number of nodes in the graph
  def get_num_node(self):
    return len(self.nodes)

  # Return a list of all node ids in the graph
  def get_all_id(self):
    all_id = [id for id in self.nodes.keys()]
    return all_id

  def contain_node(self, query_node):
    # Check if a node with the given id exists in the graph
    if query_node.id in self.nodes:
      return True
    else:
      return

  def check_node_connected(self, node1, node2):
    # Check if two nodes are connected using DFS
    if not self.contain_node(node1) or not self.contain_node(node2):
        return False
    visited = set()
    return self.dfs(node1, node2, visited)

  def dfs(self, current_node, target_node, visited):
    if current_node.id == target_node.id:
      return True
    visited.add(current_node.id)
    for neighbor, _ in current_node.edges:
      if neighbor.id not in visited:
        if self.dfs(neighbor, target_node, visited):
          return True
    return False

if __name__ == "__main__":
  import sys
  import os
  path_dir = os.path.dirname(os.path.abspath(__file__))
  from base_node import BaseNode

  base_graph = BaseGraph()
  N = 30
  for id in range(N):
    base_graph.add_node(BaseNode(id))
  base_graph.read_edge_list(os.path.join(path_dir, '../../../dataset/utils_algorithm/edge_list.txt'))
  print(base_graph)