import networkx as nx
from collections import defaultdict
from cdlib import algorithms, evaluation, NodeClustering

# File paths
graph_file = "/home/ol1ve1r4/Desktop/mocd/src/graphs/artificials/mu-0.1.edgelist"
edgelist_file = "/home/ol1ve1r4/Desktop/mocd/output.edgelist"


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
    try:
        # Load graph and GA output
        G, partition_ga = convert_edgelist_to_graph(edgelist_file)  # Convert GA output to graph and community partition

        # Louvain Algorithm
        louvain_communities = algorithms.louvain(G)

        # Leiden Algorithm
        leiden_communities = algorithms.leiden(G)

        # Compute NMI scores
        nmi_louvain = compute_nmi(partition_ga, louvain_communities, G)
        nmi_leiden = compute_nmi(partition_ga, leiden_communities, G)

        # Output results
        print(f"NMI (GA vs Louvain): {nmi_louvain:.4f}")
        print(f"NMI (GA vs Leiden): {nmi_leiden:.4f}")

        # Visualization (optional)
        pl.visualize_comparison(
            G, partition_ga, louvain_communities, nmi_louvain, "Louvain Comparison"
        )
        pl.visualize_comparison(
            G, partition_ga, leiden_communities, nmi_leiden, "Leiden Comparison"
        )

    except Exception as e:
        print(f"An error occurred: {e}")
