import re_mocd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import re_mocd, networkx as nx, matplotlib.pyplot as plt, numpy as np
from collections import defaultdict

def visualize_partition(graph, partition):
    pos, colors = nx.spring_layout(graph, seed=43), plt.cm.rainbow(np.linspace(0, 1, len(set(partition.values()))))
    color_map = {node: colors[comm] for node, comm in set(partition.items())}
    nx.draw(graph, pos, node_color=[color_map[node] for node in graph.nodes()], edge_color='gray', node_size=600, with_labels=True)
    for comm, color in enumerate(colors):
        plt.scatter([], [], c=[color], label=f'Community {comm}', s=300)
    plt.legend(loc='upper left', fontsize=10, title="Communities")
    plt.axis('off') 
    plt.savefig("example.png") 
    plt.show()

def example_graph():
    G = nx.Graph([(0, 1), (0, 3), (0, 7), (1, 2), (1, 3), (1, 5), (2, 3), (3, 6), (4, 5), (4, 6), (5, 6), (7, 8)])
    nx.write_edgelist(G, "example.edgelist", delimiter=",")
    return G

def show_example_plot():
    G = example_graph()
    partition = re_mocd.from_nx(G)
    mod = re_mocd.get_modularity(G, partition)
    visualize_partition(G, partition)
