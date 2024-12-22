import networkx as nx
from collections import defaultdict
from cdlib import algorithms, evaluation, NodeClustering
import matplotlib.pyplot as plt
import numpy as np

# File paths
graph_file = "src/graphs/artificials/mu-0.1.edgelist"
edgelist_file = "src/graphs/output/output.edgelist"


def visualize_comparison(graph, partition_ga, partition_two, nmi_score, save_file_path = None):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    pos = nx.spring_layout(graph, seed=42)

    # GA Communities Visualization
    communities_ga = defaultdict(list)
    for node, community in partition_ga.items():
        communities_ga[community].append(node)
    colors_ga = plt.cm.rainbow(np.linspace(0, 1, len(communities_ga)))
    color_map_ga = {}
    for color, nodes in zip(colors_ga, communities_ga.values()):
        for node in nodes:
            color_map_ga[node] = color
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=graph.nodes(),
                           node_color=[color_map_ga[node] for node in graph.nodes()], ax=axs[0])
    nx.draw_networkx_edges(graph, pos=pos, ax=axs[0])
    axs[0].set_title("MOCD - GA/Pareto")
    axs[0].axis('off')

    # Louvain Communities Visualization
    communities_louvain = {node: idx for idx, community in enumerate(partition_two.communities) for node in community}
    communities_louvain_dict = defaultdict(list)
    for node, community in communities_louvain.items():
        communities_louvain_dict[community].append(node)
    colors_louvain = plt.cm.rainbow(np.linspace(0, 1, len(communities_louvain_dict)))
    color_map_louvain = {}
    for color, nodes in zip(colors_louvain, communities_louvain_dict.values()):
        for node in nodes:
            color_map_louvain[node] = color
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=graph.nodes(),
                           node_color=[color_map_louvain[node] for node in graph.nodes()], ax=axs[1])
    nx.draw_networkx_edges(graph, pos=pos, ax=axs[1])
    axs[1].set_title("Second Algorithm")
    axs[1].axis('off')
    fig.suptitle(f'nmi_score: {nmi_score}', fontsize=16)
    
    if save_file_path == None:
        plt.show()
        return

    plt.savefig(save_file_path)


def compute_nmi(partition_ga, partition_algorithm, graph):
    """Compute NMI between Genetic Algorithm and another partitioning algorithm."""
    # Convert GA partition to CDLIB NodeClustering format
    communities_ga = defaultdict(list)
    for node, community in partition_ga.items():
        communities_ga[community].append(node)
    ga_communities_list = [community for community in communities_ga.values()]
    ga_node_clustering = NodeClustering(ga_communities_list, graph, "Genetic Algorithm")

    # Compute NMI
    nmi_value = evaluation.normalized_mutual_information(
        ga_node_clustering, partition_algorithm
    )
    return nmi_value.score


def convert_edgelist_to_graph(edgelist_file):
    """Convert an edgelist to a NetworkX graph and infer communities."""
    G = nx.read_edgelist(edgelist_file, delimiter=',', nodetype=int)
    # Since the edgelist does not have explicit community info,
    # we can treat each connected component as a separate community
    communities = {}
    for idx, component in enumerate(nx.connected_components(G)):
        for node in component:
            communities[node] = idx  # Assign each node in a component to a community
    return G, communities


if __name__ == "__main__":

    G, partition_ga = convert_edgelist_to_graph(edgelist_file)  # Convert GA output to graph and community partition

    louvain_communities = algorithms.louvain(G)

    leiden_communities = algorithms.leiden(G)

    nmi_louvain = compute_nmi(partition_ga, louvain_communities, G)
    nmi_leiden = compute_nmi(partition_ga, leiden_communities, G)

    print(f"NMI (GA vs Louvain): {nmi_louvain:.4f}")
    print(f"NMI (GA vs Leiden): {nmi_leiden:.4f}")

    visualize_comparison(G, partition_ga, louvain_communities, nmi_louvain)
    visualize_comparison(G, partition_ga, leiden_communities, nmi_leiden)

