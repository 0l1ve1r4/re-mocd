# Testing networkx optimization to validate my own implementation

import networkx as nx
import matplotlib.pyplot as plt
import time

time_now = time.time()

num_nodes = 3000
graph = nx.random_geometric_graph(num_nodes, 0.01)

plt.figure(figsize=(10, 10))
nx.draw(graph, node_size=5, with_labels=False)
plt.title("Random Graph with 3,000 Nodes")

time_finished = time.time() - time_now
print("Execution time:", time_finished, "seconds")

plt.show()
