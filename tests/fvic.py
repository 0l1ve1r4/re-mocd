import networkx as nx
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score
import re_mocd
from community import community_louvain  # For Louvain algorithm
import leidenalg as la  # For Leiden algorithm
import igraph as ig  # For converting networkx to iGraph

def calculate_fvic(ground_truth, detected):
    """
    Calculate the Fraction of Vertices Identified Correctly (FVIC).
    :param ground_truth: Dictionary {node: ground_truth_community}.
    :param detected: Dictionary {node: detected_community}.
    :return: FVIC value.
    """
    ground_truth_groups = defaultdict(set)
    detected_groups = defaultdict(set)
    
    for node, community in ground_truth.items():
        ground_truth_groups[community].add(node)
    for node, community in detected.items():
        detected_groups[community].add(node)
    
    fvic_sum = 0
    for detected_community in detected_groups.values():
        max_overlap = max(
            len(detected_community & ground_truth_community)
            for ground_truth_community in ground_truth_groups.values()
        )
        fvic_sum += max_overlap
    
    return fvic_sum / len(ground_truth)

def run_algorithms(graph, ground_truth):
    """
    Run the three community detection algorithms and calculate their FVIC.
    :param graph: A networkx graph object.
    :param ground_truth: Dictionary {node: ground_truth_community}.
    :return: FVIC results for each algorithm.
    """
    # RE_MOCD algorithm
    mocd = re_mocd.from_nx(graph, multi_level=False, debug=False)
    
    # Louvain algorithm
    louvain_communities = community_louvain.best_partition(graph)
    
    # Leiden algorithm
    ig_graph = ig.Graph.from_networkx(graph)
    leiden_partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
    leiden_communities = {node: membership for node, membership in enumerate(leiden_partition.membership)}
    
    # Calculate FVIC for each algorithm
    fvic_re_mocd = calculate_fvic(ground_truth, mocd)
    fvic_louvain = calculate_fvic(ground_truth, louvain_communities)
    fvic_leiden = calculate_fvic(ground_truth, leiden_communities)
    
    return {
        "RE_MOCD": fvic_re_mocd,
        "Louvain": fvic_louvain,
        "Leiden": fvic_leiden,
    }

import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
import re_mocd
from community import community_louvain
import leidenalg as la
import igraph as ig


def generate_symmetric_networks(n=1000, k=16, zin=12, zout=4):
    num_communities = 4
    size_per_community = n // num_communities
    
    # Convert degrees to probabilities for node symmetric
    p_in = zin / (size_per_community - 1)  # probability within community
    p_out = zout / (n - size_per_community) # probability between communities
    
    # Ensure probabilities are in [0,1]
    p_in = min(1.0, max(0.0, p_in))
    p_out = min(1.0, max(0.0, p_out))
    
    # Node symmetric
    node_sym = nx.random_partition_graph([size_per_community] * num_communities, p_in, p_out)
    ground_truth_node = {node: data['block'] for node, data in node_sym.nodes(data=True)}
    
    # Edge symmetric
    edge_sym = nx.planted_partition_graph(num_communities, size_per_community, p_in=p_in, p_out=p_out)
    ground_truth_edge = {node: node//size_per_community for node in range(n)}
    
    return (node_sym, ground_truth_node), (edge_sym, ground_truth_edge)

def evaluate_networks():
    zout_range = range(0, 11, 1)  # Test zout from 2 to 12
    results = {
        'node_sym': {'RE_MOCD': [], 'Louvain': [], 'Leiden': []},
        'edge_sym': {'RE_MOCD': [], 'Louvain': [], 'Leiden': []}
    }
    
    for zout in zout_range:
        node_net, edge_net = generate_symmetric_networks(zout=zout)
        
        # Evaluate node symmetric network
        node_results = run_algorithms(node_net[0], node_net[1])
        for algo in node_results:
            results['node_sym'][algo].append(node_results[algo])
            
        # Evaluate edge symmetric network
        edge_results = run_algorithms(edge_net[0], edge_net[1])
        for algo in edge_results:
            results['edge_sym'][algo].append(edge_results[algo])
    
    return list(zout_range), results

def plot_results(zout_values, results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot node symmetric results
    for algo in results['node_sym']:
        ax1.plot(zout_values, results['node_sym'][algo], marker='o', label=algo)
    ax1.set_title('Node Symmetric Network')
    ax1.set_xlabel('Zout')
    ax1.set_ylabel('FVIC')
    ax1.legend()
    ax1.grid(True)
    
    # Plot edge symmetric results
    for algo in results['edge_sym']:
        ax2.plot(zout_values, results['edge_sym'][algo], marker='o', label=algo)
    ax2.set_title('Edge Symmetric Network')
    ax2.set_xlabel('Zout')
    ax2.set_ylabel('FVIC')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('fvic_comparison.png')
    plt.show()

if __name__ == "__main__":
    zout_values, results = evaluate_networks()
    plot_results(zout_values, results)
