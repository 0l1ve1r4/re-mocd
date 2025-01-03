import re_mocd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score
import seaborn as sns
from tqdm import tqdm
import random

import community as community_louvain      # python-louvain
import igraph as ig
import leidenalg

def generate_community_graph(n_nodes=30, n_communities=3, p_in=0.3, p_out=0.05):
    """
    Generate a random graph with known community structure.
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

def louvain_communities(G, *args, **kwargs):
    """
    Detect communities using the Louvain algorithm. 
    Returns a dictionary with node: community.
    """
    # community.best_partition expects a NetworkX Graph or dict
    partition = community_louvain.best_partition(G)
    return partition

def leiden_communities(G, *args, **kwargs):
    """
    Detect communities using the Leiden algorithm (via igraph).
    Returns a dictionary with node: community.
    """
    # Convert NetworkX graph to igraph
    ig_graph = ig.Graph()
    ig_graph.add_vertices(list(G.nodes()))
    edges = list(G.edges())
    ig_graph.add_edges(edges)
    
    # Run Leiden
    partition = leidenalg.find_partition(ig_graph, leidenalg.RBConfigurationVertexPartition)
    
    # Build node: community mapping
    # partition.membership aligns with vertex order in ig_graph
    node2community = {}
    for v_idx, comm_id in enumerate(partition.membership):
        node = ig_graph.vs[v_idx]['name']  # original node index
        node2community[node] = comm_id
    return node2community

def evaluate_community_detection(algorithms, n_iterations=10, n_nodes=30, 
                                 n_communities=3, p_in=0.3, p_out=0.05):
    """
    Evaluate one or more community detection algorithms over multiple iterations.
    
    algorithms: dict 
        A dictionary of { "name_of_algorithm": function_that_returns_communities }
    """
    # Results dictionary of form: 
    # { "name_of_algorithm": { "rand_scores": [...], "community_count_accuracy": [...] }, ... }
    all_results = {name: {'rand_scores': [], 'community_count_accuracy': []}
                   for name in algorithms.keys()}
    
    for iteration in tqdm(range(n_iterations)):
        # Generate a test graph
        G, true_communities = generate_community_graph(
            n_nodes=n_nodes,
            n_communities=n_communities,
            p_in=p_in,
            p_out=p_out
        )
        # Prepare ground truth arrays
        true_labels = [true_communities[i] for i in range(len(G))]
        true_n_communities = len(set(true_communities.values()))
        
        # For each algorithm, run detection and compute metrics
        for alg_name, alg_func in algorithms.items():
            predicted_communities = alg_func(G)
            
            # Ensure all nodes have assignments
            if not all(i in predicted_communities for i in G.nodes()):
                raise ValueError(f"{alg_name} missing node assignments.")
            
            pred_labels = [predicted_communities[i] for i in range(len(G))]
            
            # Adjusted Rand Index
            ari = adjusted_rand_score(true_labels, pred_labels)
            all_results[alg_name]['rand_scores'].append(ari)
            
            # Community count accuracy (whether the number of communities matches the true count)
            pred_n_communities = len(set(predicted_communities.values()))
            count_match = (pred_n_communities == true_n_communities)
            all_results[alg_name]['community_count_accuracy'].append(count_match)
    
    return all_results

def plot_evaluation_results(all_results):
    """
    Plot the evaluation results with one or more algorithms side by side.
    """
    # Convert results into a form suitable for plotting
    # We'll build a list of (algorithm_name, ARI) for boxplot,
    # and a bar chart for community_count_accuracy.
    ari_data = []
    accuracy_data = []
    for alg_name, metrics in all_results.items():
        for score in metrics['rand_scores']:
            ari_data.append((alg_name, score))
        
        # Calculate mean accuracy for each algorithm
        bool_array = np.array(metrics['community_count_accuracy'], dtype=int)
        mean_acc = np.mean(bool_array)
        accuracy_data.append((alg_name, mean_acc))
    
    # Plot ARI with Seaborn
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=[x[0] for x in ari_data], y=[x[1] for x in ari_data])
    plt.title("Adjusted Rand Index Comparison")
    plt.ylabel("ARI")
    plt.xlabel("Algorithm")
    plt.show()
    
    # Plot Accuracy
    plt.figure(figsize=(6, 5))
    alg_names = [x[0] for x in accuracy_data]
    acc_values = [x[1] for x in accuracy_data]
    sns.barplot(x=alg_names, y=acc_values)
    plt.ylim(0, 1)
    plt.title("Community Count Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Algorithm")
    plt.show()

def run_comparison():
    algorithms = {
        "re_mocd": lambda G: re_mocd.from_nx(G, True),  
        "louvain": louvain_communities,
        "leiden":  leiden_communities,
    }
    
    all_results = evaluate_community_detection(
        algorithms,
        n_iterations=10,    
        n_nodes=100,       
        n_communities=5,
        p_in=0.3,
        p_out=0.05
    )
    
    plot_evaluation_results(all_results)

if __name__ == "__main__":
    run_comparison()