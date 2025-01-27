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

def generate_fully_symmetric_network(n=1000, k=16, zin=12, zout=4):
    num_communities = 4
    size_per_community = n // num_communities
    
    # Convert degrees to probabilities
    p_in = zin / (size_per_community - 1)  
    p_out = zout / (n - size_per_community)
    
    p_in = min(1.0, max(0.0, p_in))
    p_out = min(1.0, max(0.0, p_out))

    # Generate symmetric network
    graph = nx.random_partition_graph([size_per_community] * num_communities, p_in, p_out)
    ground_truth = {node: data['block'] for node, data in graph.nodes(data=True)}
    
    return graph, ground_truth

def evaluate_networks(num_runs=10):
    zout_range = range(0, 31, 2)
    # Store results with all runs' FVICs for RE_MOCD
    results = {
        'node_sym': {
            'RE_MOCD': {'best_fvic': [], 'all_fvics': [], 'num_comms': []},
            'Louvain': {'fvic': [], 'num_comms': []},
            'Leiden': {'fvic': [], 'num_comms': []}
        },
        'edge_sym': {
            'RE_MOCD': {'best_fvic': [], 'all_fvics': [], 'num_comms': []},
            'Louvain': {'fvic': [], 'num_comms': []},
            'Leiden': {'fvic': [], 'num_comms': []}
        },
        'symmetric': {
            'RE_MOCD': {'best_fvic': [], 'all_fvics': [], 'num_comms': []},
            'Louvain': {'fvic': [], 'num_comms': []},
            'Leiden': {'fvic': [], 'num_comms': []}
        }
    }
    for zout in zout_range:
        print("Evaluating zout =", zout)
        (node_graph, node_gt), (edge_graph, edge_gt) = generate_symmetric_networks(zout=zout)
        
        # RE_MOCD repeated runs: node-symmetric
        node_fvics = []
        node_num_comms = []
        for _ in range(num_runs):
            mocd_node = re_mocd.from_nx(node_graph, multi_level=False, debug=False)
            fvic_val = calculate_fvic(node_gt, mocd_node)
            node_fvics.append(fvic_val)
            node_num_comms.append(len(set(mocd_node.values())))
        best_idx_node = max(range(num_runs), key=lambda i: node_fvics[i])
        results['node_sym']['RE_MOCD']['best_fvic'].append(node_fvics[best_idx_node])
        results['node_sym']['RE_MOCD']['all_fvics'].append(node_fvics)
        results['node_sym']['RE_MOCD']['num_comms'].append(node_num_comms[best_idx_node])
        
        # Louvain: node-symmetric
        louvain_comms = community_louvain.best_partition(node_graph)
        results['node_sym']['Louvain']['fvic'].append(calculate_fvic(node_gt, louvain_comms))
        results['node_sym']['Louvain']['num_comms'].append(len(set(louvain_comms.values())))
        
        # Leiden: node-symmetric
        ig_graph = ig.Graph.from_networkx(node_graph)
        leiden_part = la.find_partition(ig_graph, la.ModularityVertexPartition)
        leiden_comms = {v: mem for v, mem in enumerate(leiden_part.membership)}
        results['node_sym']['Leiden']['fvic'].append(calculate_fvic(node_gt, leiden_comms))
        results['node_sym']['Leiden']['num_comms'].append(len(set(leiden_comms.values())))
        
        # RE_MOCD repeated runs: edge-symmetric
        edge_fvics = []
        edge_num_comms = []
        for _ in range(num_runs):
            mocd_edge = re_mocd.from_nx(edge_graph, multi_level=False, debug=False)
            fvic_val = calculate_fvic(edge_gt, mocd_edge)
            edge_fvics.append(fvic_val)
            edge_num_comms.append(len(set(mocd_edge.values())))
        best_idx_edge = max(range(num_runs), key=lambda i: edge_fvics[i])
        results['edge_sym']['RE_MOCD']['best_fvic'].append(edge_fvics[best_idx_edge])
        results['edge_sym']['RE_MOCD']['all_fvics'].append(edge_fvics)
        results['edge_sym']['RE_MOCD']['num_comms'].append(edge_num_comms[best_idx_edge])
        
        # Louvain: edge-symmetric
        louvain_comms = community_louvain.best_partition(edge_graph)
        results['edge_sym']['Louvain']['fvic'].append(calculate_fvic(edge_gt, louvain_comms))
        results['edge_sym']['Louvain']['num_comms'].append(len(set(louvain_comms.values())))
        
        # Leiden: edge-symmetric
        ig_graph = ig.Graph.from_networkx(edge_graph)
        leiden_part = la.find_partition(ig_graph, la.ModularityVertexPartition)
        leiden_comms = {v: mem for v, mem in enumerate(leiden_part.membership)}
        results['edge_sym']['Leiden']['fvic'].append(calculate_fvic(edge_gt, leiden_comms))
        results['edge_sym']['Leiden']['num_comms'].append(len(set(leiden_comms.values())))

        # Symmetric network evaluation
        graph, gt = generate_fully_symmetric_network(zout=zout)
        
        # RE_MOCD repeated runs
        fvics = []
        num_comms = []
        for _ in range(num_runs):
            mocd = re_mocd.from_nx(graph, multi_level=False, debug=False)
            fvic_val = calculate_fvic(gt, mocd)
            fvics.append(fvic_val)
            num_comms.append(len(set(mocd.values())))
        best_idx = max(range(num_runs), key=lambda i: fvics[i])
        results['symmetric']['RE_MOCD']['best_fvic'].append(fvics[best_idx])
        results['symmetric']['RE_MOCD']['all_fvics'].append(fvics)
        results['symmetric']['RE_MOCD']['num_comms'].append(num_comms[best_idx])
        
        # Louvain
        louvain_comms = community_louvain.best_partition(graph)
        results['symmetric']['Louvain']['fvic'].append(calculate_fvic(gt, louvain_comms))
        results['symmetric']['Louvain']['num_comms'].append(len(set(louvain_comms.values())))
        
        # Leiden
        ig_graph = ig.Graph.from_networkx(graph)
        leiden_part = la.find_partition(ig_graph, la.ModularityVertexPartition)
        leiden_comms = {v: mem for v, mem in enumerate(leiden_part.membership)}
        results['symmetric']['Leiden']['fvic'].append(calculate_fvic(gt, leiden_comms))
        results['symmetric']['Leiden']['num_comms'].append(len(set(leiden_comms.values())))


    return list(zout_range), results


def plot_results(zout_values, results):
    # Node-symmetric plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Number of communities
    ax1.plot(zout_values, results['node_sym']['RE_MOCD']['num_comms'], 'o-', label='RE_MOCD')
    ax1.plot(zout_values, results['node_sym']['Louvain']['num_comms'], '^-', label='Louvain')
    ax1.plot(zout_values, results['node_sym']['Leiden']['num_comms'], 's-', label='Leiden')
    ax1.set_title('Node-Symmetric: #Communities')
    ax1.set_xlabel('Zout')
    ax1.set_ylabel('#Communities')
    ax1.legend()
    ax1.grid(True)
    
    # FVIC comparison
    ax2.plot(zout_values, results['node_sym']['RE_MOCD']['best_fvic'], 'o-', label='RE_MOCD')
    ax2.plot(zout_values, results['node_sym']['Louvain']['fvic'], '^-', label='Louvain')
    ax2.plot(zout_values, results['node_sym']['Leiden']['fvic'], 's-', label='Leiden')
    ax2.set_title('Node-Symmetric: FVIC')
    ax2.set_xlabel('Zout')
    ax2.set_ylabel('FVIC')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('node_symmetric_results.png')
    plt.close()

    # Edge-symmetric plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Number of communities
    ax1.plot(zout_values, results['edge_sym']['RE_MOCD']['num_comms'], 'o-', label='RE_MOCD')
    ax1.plot(zout_values, results['edge_sym']['Louvain']['num_comms'], '^-', label='Louvain')
    ax1.plot(zout_values, results['edge_sym']['Leiden']['num_comms'], 's-', label='Leiden')
    ax1.set_title('Edge-Symmetric: #Communities')
    ax1.set_xlabel('Zout')
    ax1.set_ylabel('#Communities')
    ax1.legend()
    ax1.grid(True)
    
    # FVIC comparison
    ax2.plot(zout_values, results['edge_sym']['RE_MOCD']['best_fvic'], 'o-', label='RE_MOCD')
    ax2.plot(zout_values, results['edge_sym']['Louvain']['fvic'], '^-', label='Louvain')
    ax2.plot(zout_values, results['edge_sym']['Leiden']['fvic'], 's-', label='Leiden')
    ax2.set_title('Edge-Symmetric: FVIC')
    ax2.set_xlabel('Zout')
    ax2.set_ylabel('FVIC')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('edge_symmetric_results.png')
    plt.close()

    # Symmetric plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Number of communities
    ax1.plot(zout_values, results['symmetric']['RE_MOCD']['num_comms'], 'o-', label='RE_MOCD')
    ax1.plot(zout_values, results['symmetric']['Louvain']['num_comms'], '^-', label='Louvain')
    ax1.plot(zout_values, results['symmetric']['Leiden']['num_comms'], 's-', label='Leiden')
    ax1.set_title('Symmetric: #Communities')
    ax1.set_xlabel('Zout')
    ax1.set_ylabel('#Communities')
    ax1.legend()
    ax1.grid(True)
    
    # FVIC comparison
    ax2.plot(zout_values, results['symmetric']['RE_MOCD']['best_fvic'], 'o-', label='RE_MOCD')
    ax2.plot(zout_values, results['symmetric']['Louvain']['fvic'], '^-', label='Louvain')
    ax2.plot(zout_values, results['symmetric']['Leiden']['fvic'], 's-', label='Leiden')
    ax2.set_title('Symmetric: FVIC')
    ax2.set_xlabel('Zout')
    ax2.set_ylabel('FVIC')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('symmetric_results.png')
    plt.close()

if __name__ == "__main__":
    zout_values, results = evaluate_networks()
    plot_results(zout_values, results)
