import networkx as nx
from collections import defaultdict
from cdlib import algorithms, evaluation, NodeClustering
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import subprocess
import time

import rmocd

def visualize_comparison(
    graph: nx.Graph, 
    partition_ga: NodeClustering, 
    partition_two: NodeClustering, 
    nmi_score: float, 
    save_file_path: str = None
):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    pos = nx.spring_layout(graph, seed=42)

    # Visualize rmocd (partition_ga) Communities
    communities_ga = partition_ga.communities
    communities_ga_dict = defaultdict(list)
    for idx, community in enumerate(communities_ga):
        for node in community:
            communities_ga_dict[idx].append(node)
    
    colors_ga = plt.cm.rainbow(np.linspace(0, 1, len(communities_ga_dict)))
    color_map_ga = {
        node: color 
        for color, nodes in zip(colors_ga, communities_ga_dict.values()) 
        for node in nodes
    }
    nx.draw_networkx_nodes(
        graph, 
        pos=pos, 
        nodelist=graph.nodes(),
        node_color=[color_map_ga[node] for node in graph.nodes()], 
        ax=axs[0]
    )
    nx.draw_networkx_edges(graph, pos=pos, ax=axs[0])
    nx.draw_networkx_labels(graph, pos=pos, ax=axs[0])
    axs[0].set_title("rmocd - GA/Pareto")
    axs[0].axis('off')

    # Visualize the second algorithm (partition_two) Communities
    communities_algo = partition_two.communities
    communities_algo_dict = defaultdict(list)
    for idx, community in enumerate(communities_algo):
        for node in community:
            communities_algo_dict[idx].append(node)
    
    colors_algo = plt.cm.rainbow(np.linspace(0, 1, len(communities_algo_dict)))
    color_map_algo = {
        node: color 
        for color, nodes in zip(colors_algo, communities_algo_dict.values()) 
        for node in nodes
    }
    nx.draw_networkx_nodes(
        graph, 
        pos=pos, 
        nodelist=graph.nodes(),
        node_color=[color_map_algo[node] for node in graph.nodes()], 
        ax=axs[1]
    )
    nx.draw_networkx_edges(graph, pos=pos, ax=axs[1])
    nx.draw_networkx_labels(graph, pos=pos, ax=axs[1])
    axs[1].set_title("Second Algorithm (Louvain/Leiden)")
    axs[1].axis('off')

    fig.suptitle(f'NMI Score: {nmi_score:.4f}', fontsize=16)

    if save_file_path is None:
        plt.show()
    else:
        plt.savefig(save_file_path)

def compute_nmi(partition_ga: dict, partition_algorithm: NodeClustering, graph: nx.Graph):
    """Compute NMI between Genetic Algorithm partition (dictionary) and another partitioning algorithm."""
    communities_ga = defaultdict(list)
    for node, community in partition_ga.items():
        communities_ga[community].append(node)
    ga_communities_list = list(communities_ga.values())
    ga_node_clustering = NodeClustering(ga_communities_list, graph, "Genetic Algorithm")

    nmi_value = evaluation.normalized_mutual_information(ga_node_clustering, partition_algorithm)
    return nmi_value.score

def convert_edgelist_to_graph(edgelist_file: str):
    """Convert an edgelist to a NetworkX graph."""
    try:
        G = nx.read_edgelist(edgelist_file, delimiter=',', nodetype=int)
        return G
    except Exception as e:
        print(f"Error reading edgelist file: {e}")
        raise

def convert_to_node_clustering(partition_dict, graph):
    """Convert a dictionary partition to NodeClustering."""
    communities = defaultdict(list)
    for node, community in partition_dict.items():
        communities[community].append(node)

    community_list = list(communities.values())
    return NodeClustering(community_list, graph, "rmocd Algorithm")

def run_comparisons(graph_file: str, show_plot: bool):
    # Run the rmocd approach

    start = time.time()
    mocd_partition, modularity = rmocd.run(graph_file)

    if show_plot:
        print(f"rmocd modularity: {modularity}")
        print(f"Time spent: {time.time() - start}")

    # Read the graph
    G = convert_edgelist_to_graph(graph_file)

    # Convert rmocd partition (dict) to NodeClustering
    mocd_nc = convert_to_node_clustering(mocd_partition, G)

    # Run Louvain and Leiden
    louvain_communities = algorithms.louvain(G)
    leiden_communities = algorithms.leiden(G)

    print(f"Louvain communities: {louvain_communities}")
    print(f"Leiden communities: {leiden_communities}")

    # Compute NMI
    nmi_louvain = compute_nmi(mocd_partition, louvain_communities, G)
    nmi_leiden = compute_nmi(mocd_partition, leiden_communities, G)

    # Visualize comparisons
    visualize_comparison(G, mocd_nc, louvain_communities, nmi_louvain, "output")
    visualize_comparison(G, mocd_nc, leiden_communities, nmi_leiden, "output")

mu_graphs = [f"res/graphs/artificials/mu-0.{i}.edgelist" for i in range(1, 9)]

if __name__ == "__main__":
    runs_per_file = 10
    
    has_args = (len(sys.argv) > 1)
    graph_files = None
    num_files = None
    show_plot = False

    if has_args:
        graph_files = (sys.argv[1:])[0]
        num_files = len(sys.argv[1:])
    else: 
        graph_files = mu_graphs
        num_files = len(graph_files)

    if num_files == 1:
        show_plot = True
        run_comparisons(graph_files, show_plot)
    else:
        for i in range(num_files):
            for _ in range(runs_per_file):
                run_comparisons(graph_files[i], show_plot)

    print("Done.")