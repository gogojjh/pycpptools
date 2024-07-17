class BaseGraph:
  def __init__(self):
    # Initialize an empty dictionary to store nodes
    self.nodes = {}

  def add_node(self, new_node):
    # Add a new node to the graph if it doesn't already exist
    if new_node.id not in self.nodes:
      self.nodes[new_node.id] = new_node

  def add_edge(self, from_node, to_node, weight):
    # Add an edge between two nodes if both nodes exist in the graph
    if from_node.id in self.nodes and to_node.id in self.nodes:
      from_node.add_edge(to_node, weight)
      to_node.add_edge(from_node, weight)  # Assuming undirected graph

  def get_node(self, id):
    # Return the node with the given id if it exists, otherwise return None
    if id in self.nodes:
      return self.nodes[id]
    else:
      return None

  def get_num_node(self):
    # Return the number of nodes in the graph
    return len(self.nodes)

  def get_all_id(self):
    # Return a list of all node ids in the graph
    all_id = [id for id in self.nodes.keys()]
    return all_id

  def contain_node(self, query_node):
    # Check if a node with the given id exists in the graph
    if query_node.id in self.nodes:
      return True
    else:
      return
