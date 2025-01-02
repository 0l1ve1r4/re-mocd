from cdlib import algorithms, evaluation, NodeClustering
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import subprocess
import re_mocd
import json
import time

def convert_to_node_clustering(partition_dict, graph):
    """Convert a dictionary partition to NodeClustering."""
    communities = defaultdict(list)
    for node, community in partition_dict.items():
        communities[community].append(node)

    community_list = list(communities.values())
    return NodeClustering(community_list, graph, "re_mocd Algorithm")

def compute_nmi(partition_ga: dict, partition_algorithm: NodeClustering, graph: nx.Graph):
    """Compute NMI between Genetic Algorithm partition (dictionary) and another partitioning algorithm."""
    communities_ga = defaultdict(list)
    for node, community in partition_ga.items():
        communities_ga[community].append(node)
    ga_communities_list = list(communities_ga.values())
    ga_node_clustering = NodeClustering(ga_communities_list, graph, "Genetic Algorithm")

    nmi_value = evaluation.normalized_mutual_information(ga_node_clustering, partition_algorithm)
    return nmi_value.score

def visualize_comparison(graph: nx.Graph, partition_ga: NodeClustering, partition_two: NodeClustering, nmi_score: float, save_file_path: str = None, title: str = "Algorithm"):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    pos = nx.spring_layout(graph)
    
    def plot_partition(community_dict, ax, title):
        colors = plt.cm.Set3(np.linspace(0, 1, len(community_dict)))
        color_map = {node: color for color, nodes in zip(colors, community_dict.values()) for node in nodes}
        
        nx.draw_networkx_nodes(graph, pos=pos, node_color=[color_map[node] for node in graph.nodes()], node_size=500, ax=ax)
        nx.draw_networkx_edges(graph, pos=pos, ax=ax, edge_color='gray', width=1.0, alpha=0.5)
        nx.draw_networkx_labels(graph, pos=pos, ax=ax, font_size=8, font_weight='bold')
        ax.set_title(title, pad=20)
        ax.axis('off')
    
    # Plot both partitions
    plot_partition({idx: list(community) for idx, community in enumerate(partition_ga.communities)}, axs[0], "re_mocd")
    plot_partition({idx: list(community) for idx, community in enumerate(partition_two.communities)}, axs[1], title)
    
    # Add NMI score
    fig.suptitle(f'NMI Score: {nmi_score:.4f}', fontsize=16, y=0.95)
    plt.tight_layout(pad=3.0)
    
    if save_file_path:
        plt.savefig(save_file_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def run_comparisons(G, show_plot: bool):
    start = time.time()
    mocd_partition = re_mocd.from_nx(G)
    if show_plot:
        print(f"Elapsed: {time.time() - start}")
    mocd_nc = convert_to_node_clustering(mocd_partition, G)

    louvain_communities = algorithms.louvain(G)
    leiden_communities = algorithms.leiden(G)

    nmi_louvain = compute_nmi(mocd_partition, louvain_communities, G)
    nmi_leiden = compute_nmi(mocd_partition, leiden_communities, G)

    if show_plot:
        visualize_comparison(G, mocd_nc, louvain_communities, nmi_louvain, "output_louvain", title="louvain")
        visualize_comparison(G, mocd_nc, leiden_communities, nmi_leiden, "output_leiden", title="Leiden")