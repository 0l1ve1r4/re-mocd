import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.widgets import Button
import re_mocd
import pandas as pd
from sklearn.cluster import DBSCAN

def visualize_partition(graph, partition):
    """
    Visualizes the graph with communities more distinctly separated.
    Allows clicking on a node to highlight its community by removing other nodes
    and provides a button to restore the original graph.
    
    Parameters:
        graph (nx.Graph): The graph to visualize.
        partition (dict): A dictionary where keys are nodes and values are community indices.
    """
    # Generate layout
    pos = nx.spring_layout(graph, seed=43, k=1.0, iterations=100)

    # Determine communities and their colors
    communities = sorted(set(partition.values()))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))
    color_map = {comm: color for comm, color in zip(communities, colors)}
    node_colors = [color_map[partition[node]] for node in graph.nodes()]

    # Store the original graph for restoration
    original_nodes = list(graph.nodes())
    original_edges = list(graph.edges())

    # Initial draw of the graph
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(
        graph, pos, node_color=node_colors, edge_color='gray',
        node_size=600, with_labels=True, ax=ax
    )

    # Add legend
    for comm in communities:
        ax.scatter([], [], c=[color_map[comm]], label=f'Community {comm}', s=300)
    ax.legend(loc='upper left', fontsize=10, title="Communities")
    plt.axis('off')

    # Function to handle click events
    def on_click(event):
        if event.inaxes == ax:
            # Identify the closest node to the click
            distances = {node: np.linalg.norm(pos[node] - [event.xdata, event.ydata]) for node in graph.nodes()}
            clicked_node = min(distances, key=distances.get)

            # Determine the community of the clicked node
            clicked_community = partition[clicked_node]

            # Filter the graph to include only nodes in the clicked community
            nodes_to_keep = [node for node in graph.nodes() if partition[node] == clicked_community]
            subgraph = graph.subgraph(nodes_to_keep)

            # Clear the plot and redraw the subgraph
            ax.clear()
            nx.draw(
                subgraph, pos, node_color=[color_map[clicked_community]] * len(nodes_to_keep),
                edge_color='gray', node_size=600, with_labels=True, ax=ax
            )

            # Redraw the legend
            ax.scatter([], [], c=[color_map[clicked_community]], label=f'Community {clicked_community}', s=300)
            ax.legend(loc='upper left', fontsize=10, title="Community Highlighted")
            plt.axis('off')
            fig.canvas.draw_idle()

    # Function to restore the original graph
    def restore(event):
        graph.clear()
        graph.add_nodes_from(original_nodes)
        graph.add_edges_from(original_edges)

        # Clear the plot and redraw the original graph
        ax.clear()
        nx.draw(
            graph, pos, node_color=node_colors, edge_color='gray',
            node_size=600, with_labels=True, ax=ax
        )

        # Redraw the legend
        for comm in communities:
            ax.scatter([], [], c=[color_map[comm]], label=f'Community {comm}', s=300)
        ax.legend(loc='upper left', fontsize=10, title="Communities")
        plt.axis('off')
        fig.canvas.draw_idle()

    # Add a restore button
    restore_ax = plt.axes([0.8, 0.01, 0.1, 0.05])  # Position: [left, bottom, width, height]
    restore_button = Button(restore_ax, 'Restore')
    restore_button.on_clicked(restore)

    # Connect the click event to the handler
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Show the plot
    plt.show()

def create_filtered_graph(url):
    """
    Filters a graph's edges based on clustering (DBSCAN).
    
    :param url: The URL of the CSV file.
    :param eps: Maximum distance between two samples for them to be considered in the same neighborhood.
    :param min_samples: Minimum number of edges required to form a cluster.
    :return: A filtered undirected, unweighted NetworkX Graph.
    """
    # Read the CSV
    df = pd.read_csv(url)
    
    edges = df[['Source', 'Target']].values
    weights = df['weight'].values
    
    weights_reshaped = weights.reshape(-1, 1)
    
    clustering = DBSCAN(min_samples=2).fit(weights_reshaped)
    
    df['cluster'] = clustering.labels_
    df_filtered = df[df['cluster'] != -1]
    
    G = nx.from_pandas_edgelist(
        df_filtered,
        source='Source',
        target='Target'
    )
    
    return G

def transform_graph_to_numbers(G):
    """
    Transforms a graph with string nodes into a graph with numeric nodes.
    Returns the transformed graph and a mapping from original to numeric nodes.
    """
    node_map = {node: idx for idx, node in enumerate(G.nodes())}
    G_numeric = nx.relabel_nodes(G, node_map)
    return G_numeric, {v: k for k, v in node_map.items()}

def transform_partition_to_original(partition, mapping):
    """
    Transforms a partition from numeric nodes back to original nodes.
    """
    return {mapping[node]: community for node, community in partition.items()}

def run():
    G = create_filtered_graph("tests/python/benchmarks/got.csv")
    G_numeric, mapping = transform_graph_to_numbers(G)
    partition_numeric = re_mocd.from_nx(G_numeric, multi_level=True, debug=True)
    partition = transform_partition_to_original(partition_numeric, mapping)
    visualize_partition(G, partition)

if __name__ == "__main__":
    run()