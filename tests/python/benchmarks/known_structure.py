# Evaluate the re_mocd detection algorithm over multiple iterations in known structures

import re_mocd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
from tqdm import tqdm
import random

def generate_community_graph(n_nodes=30, n_communities=3, p_in=0.3, p_out=0.05):
    """
    Generate a random graph with known community structure
    
    Parameters:
    -----------
    n_nodes : int
        Total number of nodes
    n_communities : int
        Number of communities to generate
    p_in : float
        Probability of intra connection 
    p_out : float
        Probability of inter connection
    
    Returns:
    --------
    G : networkx.Graph
        Generated graph
    true_communities : dict
        Dictionary mapping node IDs to their true community assignments
    """
    nodes_per_community = n_nodes // n_communities
    remainder = n_nodes % n_communities
    
    community_sizes = [nodes_per_community] * n_communities
    for i in range(remainder):
        community_sizes[i] += 1
    
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    
    # Assign nodes to communities
    true_communities = {}
    start_idx = 0
    for comm_id, size in enumerate(community_sizes):
        for node in range(start_idx, start_idx + size):
            true_communities[node] = comm_id
        start_idx += size
    
    # Add edges based on community structure
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if true_communities[i] == true_communities[j]:
                if random.random() < p_in:
                    G.add_edge(i, j)
            else:
                if random.random() < p_out:
                    G.add_edge(i, j)
    
    return G, true_communities

def evaluate_community_detection(your_algorithm, n_iterations=100, n_nodes=30, 
                              n_communities=3, p_in=0.3, p_out=0.05):
    """
    Evaluate the community detection algorithm over multiple iterations
    """
    rand_scores = []
    community_count_accuracy = []
    
    for iteration in tqdm(range(n_iterations)):
        # Generate a test graph
        G, true_communities = generate_community_graph(
            n_nodes=n_nodes,
            n_communities=n_communities,
            p_in=p_in,
            p_out=p_out
        )
        
        # Run the algorithm
        predicted_communities = your_algorithm(G)
        
        # Debug information
        if not all(i in predicted_communities for i in range(len(G))):
            missing_nodes = [i for i in range(len(G)) if i not in predicted_communities]
            print(f"\nIteration {iteration}")
            print(f"Missing nodes in predicted communities: {missiing_nodes}")            
            raise ValueError("Predicted communities dictionary is missing some nodes!")
        
        # Calculate metrics
        try:
            true_labels = [true_communities[i] for i in range(len(G))]
            pred_labels = [predicted_communities[i] for i in range(len(G))]
            
            # Calculate adjusted Rand index
            ari = adjusted_rand_score(true_labels, pred_labels)
            rand_scores.append(ari)
            
            # Check if number of communities is correct
            true_n_communities = len(set(true_communities.values()))
            pred_n_communities = len(set(predicted_communities.values()))
            community_count_accuracy.append(true_n_communities == pred_n_communities)
            
        except KeyError as e:
            print(f"\nKeyError in iteration {iteration}")
            print(f"Missing key: {e}")
            print(f"Graph nodes: {list(G.nodes())}")
            print(f"Predicted communities keys: {list(predicted_communities.keys())}")
            raise
    
    return {
        'rand_scores': rand_scores,
        'community_count_accuracy': community_count_accuracy
    }

def plot_evaluation_results(results):
    """
    Plot the evaluation results with confidence intervals
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Adjusted Rand Index distribution
    sns.boxplot(y=results['rand_scores'], ax=ax1)
    ax1.set_title('Distribution of Adjusted Rand Index')
    ax1.set_ylabel('Adjusted Rand Index')
    
    # Add 95% confidence interval for ARI
    mean_ari = np.mean(results['rand_scores'])
    ci = np.percentile(results['rand_scores'], [2.5, 97.5])
    ax1.axhline(y=mean_ari, color='r', linestyle='--', label=f'Mean: {mean_ari:.3f}')
    ax1.axhspan(ci[0], ci[1], alpha=0.2, color='r', label='95% CI')
    ax1.legend()
    
    # Convert boolean array to integers for accuracy calculation
    accuracy_values = np.array(results['community_count_accuracy']).astype(int)
    accuracy = np.mean(accuracy_values)
    
    # Calculate confidence interval for accuracy using integers
    ci_accuracy = np.percentile(accuracy_values, [2.5, 97.5])
    
    ax2.bar(['Accuracy'], [accuracy])
    ax2.errorbar(['Accuracy'], [accuracy], 
                 yerr=[[accuracy-ci_accuracy[0]], [ci_accuracy[1]-accuracy]], 
                 fmt='none', color='black', capsize=5)
    ax2.set_title('Community Count Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def run(subprocess: bool = False):
    results = []
    results = evaluate_community_detection(
        re_mocd.from_nx,
        n_iterations=10,
        n_nodes=100,
        n_communities=5,
        p_in=0.3,
        p_out=0.05
    )

    plot_evaluation_results(results)