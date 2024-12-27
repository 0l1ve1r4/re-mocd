import rmocd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def visualize_partition(graph: nx.Graph, partition_dict: dict):
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)

    # Visualize communities from partition_dict
    communities_dict = defaultdict(list)
    for node, community in partition_dict.items():
        communities_dict[community].append(node)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(communities_dict)))
    color_map = {
        node: color 
        for color, nodes in zip(colors, communities_dict.values()) 
        for node in nodes
    }

    nx.draw_networkx_edges(graph, pos=pos, ax=ax, alpha=0.5)
    nx.draw_networkx_nodes(
        graph, 
        pos=pos, 
        nodelist=graph.nodes(),
        node_color=[color_map[node] for node in graph.nodes()], 
        node_size=100,
        ax=ax
    )
    nx.draw_networkx_labels(graph, pos=pos, ax=ax, font_size=8)
    
    # Creating a legend
    for idx, (community, color) in enumerate(zip(communities_dict.values(), colors)):
        ax.scatter([], [], c=[color], label=f'Community {idx}', s=100)
    
    ax.legend(scatterpoints=1, loc='best', fontsize=8)
    ax.axis('off')

    plt.savefig("Example.png")
    plt.show()

edgelist_file = "res/graphs/artificials/karate.edgelist"
G = nx.read_edgelist(edgelist_file, delimiter=',', nodetype=int)

mocd_partition, modularity = rmocd.run(edgelist_file, infinity=True)
visualize_partition(G, mocd_partition)