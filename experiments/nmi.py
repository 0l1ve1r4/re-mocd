import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.generators.community import LFR_benchmark_graph
from re_mocd import rmocd
from community import community_louvain
import leidenalg as la
import igraph as ig
from math import log
import time

def generate_lfr_benchmark(mu, n=500, tau1=2, tau2=1.5, max_d=15, min_comm=20, max_comm=50):
    """
    Generate LFR benchmark network
    
    Parameters:
    - mu: mixing parameter (between 0 and 1)
    - n: number of nodes
    - tau1: node degree distribution power law exponent
    - tau2: community size distribution power law exponent
    - avg_d: average node degree
    - max_d: maximum node degree
    - min_comm: minimum community size
    - max_comm: maximum community size
    """
    try:
        G = LFR_benchmark_graph(
            n=n,
            tau1=tau1,
            tau2=tau2,
            mu=mu,
            max_degree=max_d,
            average_degree=5,
            min_community=min_comm,
            max_community=max_comm,
            seed=42  # for reproducibility
        )
        
        # Get ground truth communities
        ground_truth = {node: next(iter(communities)) 
                       for node, communities in G.nodes(data='community')}
        
        return G, ground_truth
    except Exception as e:
        print(f"Error generating LFR benchmark: {e}")
        return None, None

def calculate_nmi(detected_communities, ground_truth):
    """
    Calcula Normalized Mutual Information entre duas partições
    """
    if not detected_communities or not ground_truth:
        return 0.0
        
    N = float(len(ground_truth))    
    detected_counts = {}
    true_counts = {}
    confusion = {}
    
    for i in detected_communities.values():
        if i not in detected_counts:
            detected_counts[i] = 0
            confusion[i] = {}
        detected_counts[i] += 1
        
    for i in ground_truth.values():
        if i not in true_counts:
            true_counts[i] = 0
        true_counts[i] += 1
        for j in confusion:
            confusion[j][i] = 0
            
    for node in ground_truth:
        if node in detected_communities:
            i = detected_communities[node]
            j = ground_truth[node]
            confusion[i][j] += 1
    
    mi = 0.0
    for i in confusion:
        for j in confusion[i]:
            if confusion[i][j] > 0:
                mi += (confusion[i][j] / N) * log((N * confusion[i][j]) / 
                                                (detected_counts[i] * true_counts[j]), 2)
    
    # entropy
    h_detected = 0.0
    for count in detected_counts.values():
        if count > 0:
            h_detected += -(count / N) * log(count / N, 2)
    
    h_true = 0.0
    for count in true_counts.values():
        if count > 0:
            h_true += -(count / N) * log(count / N, 2)
    
    if h_detected == 0 or h_true == 0:
        return 0.0
    
    return 2.0 * mi / (h_detected + h_true)


def run_algorithms(G, ground_truth):
    """Executa algoritmos disponíveis de detecção de comunidades e mede o tempo"""
    results = {}
    times = {}
        
    # MOCD
    
    start_time = time.time()
    fmocd_communities = rmocd(G, 50)
    times['rmocd'] = time.time() - start_time
    results['rmocd'] = calculate_nmi(fmocd_communities, ground_truth)
    
    # Louvain
    try:
        start_time = time.time()
        communities_louvain = {node: comm for node, comm 
                             in community_louvain.best_partition(G).items()}
        times['Louvain'] = time.time() - start_time
        results['Louvain'] = calculate_nmi(communities_louvain, ground_truth)
    except Exception as e:
        print(f"Erro Louvain: {e}")
        results['Louvain'] = 0
        times['Louvain'] = 0
    
    # Leiden
    try:
        start_time = time.time()
        G_ig = ig.Graph(edges=list(G.edges()), n=G.number_of_nodes())
        partition = la.find_partition(G_ig, la.ModularityVertexPartition)
        communities_leiden = {node: comm for node, comm in enumerate(partition.membership)}
        times['Leiden'] = time.time() - start_time
        results['Leiden'] = calculate_nmi(communities_leiden, ground_truth)
    except Exception as e:
        print(f"Erro Leiden: {e}")
        results['Leiden'] = 0
        times['Leiden'] = 0
    
    return results, times

# Modified experiment parameters
mu_values = np.linspace(0.0, 1.0, 21)
n_runs = 10
algorithms = ['rmocd', 'Louvain', 'Leiden']
colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 'o', 'o', 'o', 'o', 'o']

# Initialize results dictionaries
results = {alg: {'mean': [], 'std': []} for alg in algorithms}
execution_times = {alg: {'mean': [], 'std': []} for alg in algorithms}

# Run experiments
for mu in mu_values:
    print(f"Processing μ = {mu:.2f}")
    runs_results = {alg: [] for alg in algorithms}
    runs_times = {alg: [] for alg in algorithms}
    
    for run in range(n_runs):
        G, ground_truth = generate_lfr_benchmark(mu=mu)
        if G is not None:
            run_results, run_times = run_algorithms(G, ground_truth)
            
            for alg in algorithms:
                runs_results[alg].append(run_results[alg])
                runs_times[alg].append(run_times[alg])
    
    # Calculate statistics
    for alg in algorithms:
        results[alg]['mean'].append(np.mean(runs_results[alg]))
        results[alg]['std'].append(np.std(runs_results[alg]))
        execution_times[alg]['mean'].append(np.mean(runs_times[alg]))
        execution_times[alg]['std'].append(np.std(runs_times[alg]))

# Create figure with three subplots
fig = plt.figure(figsize=(18, 6))

# Plot 1: All algorithms NMI comparison
ax1 = fig.add_subplot(131)
for alg, color, marker in zip(algorithms, colors, markers):
    ax1.errorbar(mu_values, 
                results[alg]['mean'], 
                yerr=results[alg]['std'],
                label=alg,
                color=color,
                marker=marker,
                capsize=3,
                markersize=8,
                linewidth=2)
ax1.set_xlabel('μ', fontsize=12)
ax1.set_ylabel('NMI', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(fontsize=10)
ax1.set_ylim(0, 1.05)

plt.tight_layout()
plt.show()