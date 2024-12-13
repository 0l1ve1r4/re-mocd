import networkx as nx
import random
import matplotlib.pyplot as plt

import genetic_algorithm as GA

def visualize_partition(graph, partition):
    """Visualize the graph with partitioned communities in one window."""
    communities = {}
    for node, community in partition.items():
        communities.setdefault(community, []).append(node)

    # Helper function to get the edge list without edges between communities
    def remove_inter_community_edges(graph, partition):
        edges_to_remove = []
        for u, v in graph.edges:
            if partition[u] != partition[v]:
                edges_to_remove.append((u, v))
        graph_copy = graph.copy()
        graph_copy.remove_edges_from(edges_to_remove)
        return graph_copy

    # Create a figure for the subplots
    plt.figure(figsize=(18, 6))

    # 1. Complete graph visualization
    plt.subplot(1, 3, 1)  # 1 row, 3 columns, first subplot
    nx.draw(graph, with_labels=True, node_color='lightgray', edge_color='gray')
    plt.title("Complete Graph")

    # 2. Colorized graph with communities
    color_map = {}
    colors = plt.cm.tab10(range(len(communities)))

    for color, nodes in zip(colors, communities.values()):
        for node in nodes:
            color_map[node] = color

    node_colors = [color_map[node] for node in graph.nodes]
    plt.subplot(1, 3, 2)  # 1 row, 3 columns, second subplot
    nx.draw(graph, with_labels=True, node_color=node_colors, edge_color='gray')
    plt.title("Graph with Communities")

    # 3. Colorized graph with communities, without edges between communities
    graph_no_inter_edges = remove_inter_community_edges(graph, partition)
    node_colors_no_inter = [color_map[node] for node in graph_no_inter_edges.nodes]
    plt.subplot(1, 3, 3)  # 1 row, 3 columns, third subplot
    nx.draw(graph_no_inter_edges, with_labels=True, node_color=node_colors_no_inter, edge_color='gray')
    plt.title("Communities Without Edges Between Communities")

    # Show all plots in the same window
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    G = nx.Graph()
    edges = [(0,1), (0,2), (0,3), (0,7), 
             (1,2), (1,3), (1,5), (1,7),
             (2,3),
             (4, 5), 
             (5, 6), 
             (6, 4), 
             (6, 8), 
             (7, 8)]

    G.add_edges_from(edges)

    print("Calling GA")
    best_partition = GA.genetic_algorithm(G, population_size=100, generations=80)
    visualize_partition(G, best_partition)

    print("The communities are:")
    communities = {}
    for node, community in best_partition.items():
        communities.setdefault(community, []).append(node)

    for community in communities.values():
        print(", ".join(map(str, community)))
