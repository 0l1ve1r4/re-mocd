# Testing networkx optimization to validate my own implementation

import networkx as nx
import matplotlib.pyplot as plt

# Create a random graph with 30,000 nodes
num_nodes = 30000
graph = nx.random_geometric_graph(num_nodes, 0.01)

# Draw the graph
plt.figure(figsize=(10, 10))
nx.draw(graph, node_size=5, with_labels=False)
plt.title("Random Graph with 30,000 Nodes")
plt.show()
