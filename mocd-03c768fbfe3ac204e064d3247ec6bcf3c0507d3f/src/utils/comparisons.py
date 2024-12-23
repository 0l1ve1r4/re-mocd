import networkx as nx
from collections import defaultdict
from cdlib import algorithms, evaluation, NodeClustering
import matplotlib.pyplot as plt
import numpy as np
import json

# File paths
graph_file = "src/graphs/artificials/karate.edgelist"
best_partition_json = "src/graphs/output/output.json"

def visualize_comparison(graph: nx.Graph, partition_ga: dict, partition_two: NodeClustering, nmi_score: float, save_file_path: str = None):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    pos = nx.spring_layout(graph, seed=42)

    # GA Communities Visualization
    communities_ga = defaultdict(list)
    for node, community in partition_ga.items():
        communities_ga[community].append(node)
    colors_ga = plt.cm.rainbow(np.linspace(0, 1, len(communities_ga)))
    color_map_ga = {node: color for color, nodes in zip(colors_ga, communities_ga.values()) for node in nodes}
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=graph.nodes(),
                           node_color=[color_map_ga[node] for node in graph.nodes()], ax=axs[0])
    nx.draw_networkx_edges(graph, pos=pos, ax=axs[0])
    axs[0].set_title("MOCD - GA/Pareto")
    axs[0].axis('off')

    # Second Algorithm (Louvain/Leiden) Communities Visualization
    communities_algo = {node: idx for idx, community in enumerate(partition_two.communities) for node in community}
    communities_algo_dict = defaultdict(list)
    for node, community in communities_algo.items():
        communities_algo_dict[community].append(node)
    colors_algo = plt.cm.rainbow(np.linspace(0, 1, len(communities_algo_dict)))
    color_map_algo = {node: color for color, nodes in zip(colors_algo, communities_algo_dict.values()) for node in nodes}
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=graph.nodes(),
                           node_color=[color_map_algo[node] for node in graph.nodes()], ax=axs[1])
    nx.draw_networkx_edges(graph, pos=pos, ax=axs[1])
    axs[1].set_title("Second Algorithm (Louvain/Leiden)")
    axs[1].axis('off')
    fig.suptitle(f'NMI Score: {nmi_score:.4f}', fontsize=16)
    
    if save_file_path is None:
        plt.show()
    else:
        plt.savefig(save_file_path)

def compute_nmi(partition_ga: dict, partition_algorithm: NodeClustering, graph: nx.Graph):
    """Compute NMI between Genetic Algorithm and another partitioning algorithm."""
    # Convert GA partition to CDLIB NodeClustering format
    communities_ga = defaultdict(list)
    for node, community in partition_ga.items():
        communities_ga[community].append(node)
    ga_communities_list = list(communities_ga.values())
    ga_node_clustering = NodeClustering(ga_communities_list, graph, "Genetic Algorithm")

    # Compute NMI
    nmi_value = evaluation.normalized_mutual_information(
        ga_node_clustering, partition_algorithm
    )
    return nmi_value.score

def convert_edgelist_to_graph(edgelist_file: str):
    """Convert an edgelist to a NetworkX graph."""
    try:
        G = nx.read_edgelist(edgelist_file, delimiter=',', nodetype=int)
        return G
    except Exception as e:
        print(f"Error reading edgelist file: {e}")
        raise

def load_best_partition(file_path):
    """Load the best partition from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {int(k): v for k, v in data.items()}

if __name__ == "__main__":
    try:
        # Convert the graph from the edgelist file
        G = convert_edgelist_to_graph(graph_file)
        best_partition = load_best_partition(best_partition_json)

        # Apply Louvain and Leiden algorithms (using CDLib)
        louvain_communities = algorithms.louvain(G)
        leiden_communities = algorithms.leiden(G)

        # Compute NMI between GA and Louvain/Leiden partitions
        nmi_louvain = compute_nmi(best_partition, louvain_communities, G)
        nmi_leiden = compute_nmi(best_partition, leiden_communities, G)

        # Print the NMI scores
        print(f"NMI (GA vs Louvain): {nmi_louvain:.4f}")
        print(f"NMI (GA vs Leiden): {nmi_leiden:.4f}")


    except Exception as e:
        print(f"An error occurred: {e}")

            # Visualize the comparison of community partitions
    visualize_comparison(G, best_partition, louvain_communities, nmi_louvain)
    visualize_comparison(G, best_partition, leiden_communities, nmi_leiden)
    print("\nDone.")