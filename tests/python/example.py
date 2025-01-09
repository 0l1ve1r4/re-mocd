import re_mocd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

def from_csv(file_path):
    """
    Creates a NetworkX graph from a CSV file with specified headers.
    Only includes edges from 'src' to 'trg' without weights.
    """
    df = pd.read_csv(file_path)
    G = nx.Graph()
    G.add_edges_from(df[['src', 'trg']].values)
    return G

def example_graph():
    G = nx.Graph([(0, 1), (0, 3), (0, 7), (1, 2), (1, 3), (1, 5), (2, 3), (3, 6), (4, 5), (4, 6), (5, 6), (7, 8)])
    nx.write_edgelist(G, "example.edgelist", delimiter=",")
    return G

def calculate_modularity(G, partition):
    """
    Calculate modularity of a graph partition.
    """
    # Convert partition to a format suitable for NetworkX
    community_dict = {}
    for node, comm in partition.items():
        community_dict.setdefault(comm, []).append(node)
    communities = list(community_dict.values())
    
    # Calculate modularity using NetworkX's built-in function
    modularity = nx.algorithms.community.modularity(G, communities)
    
    return modularity

def show_example_plot():
    G = nx.karate_club_graph()
    #G = from_csv("tests/python/RIOTS-edgelist.csv")
    partition = re_mocd.from_nx(G, multi_level=True, debug=True)
    print(f"modularity: {calculate_modularity(G, partition)}")
    visualize_partition(G, partition)
