import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import re_mocd
import random
import time

from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import normalized_mutual_info_score
from collections import defaultdict
from scipy import stats

class CommunityMetrics:
    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
    
    def add_metric(self, name: str, value: float):
        self.metrics[name].append(value)
    
    def get_confidence_interval(self, metric: str):
        """Returns mean and 95% confidence interval for a metric"""
        values = np.array(self.metrics[metric])
        mean = np.mean(values)
        
        if len(values) < 2:
            return mean, mean, mean
            
        std_err = stats.sem(values)
        ci = stats.t.interval(confidence=0.95, 
                            df=len(values)-1,
                            loc=mean,
                            scale=std_err)
        
        return mean, ci[0], ci[1]

def validate_communities(
    G: nx.Graph,
    partitions: Dict[int, int],
    ground_truth: Optional[Dict[int, int]] = None
):
    """Validates the quality of found community partitions"""
    metrics = {}
    
    try:
        communities = defaultdict(list)
        for node, comm_id in partitions.items():
            communities[comm_id].append(node)
        
        # Modularity
        metrics['modularity'] = nx.community.modularity(G, communities.values())
        
        # Internal density
        internal_density = 0
        for comm_nodes in communities.values():
            subgraph = G.subgraph(comm_nodes)
            possible_edges = len(comm_nodes) * (len(comm_nodes) - 1) / 2
            if possible_edges > 0:
                density = subgraph.number_of_edges() / possible_edges
                internal_density += density
                
        metrics['internal_density'] = internal_density / len(communities)
        
        # NMI for ground truth comparison
        if ground_truth:
            true_labels = [ground_truth[n] for n in G.nodes()]
            pred_labels = [partitions[n] for n in G.nodes()]
            metrics['nmi'] = normalized_mutual_info_score(true_labels, pred_labels)
            
        metrics['n_communities'] = len(communities)
        comm_sizes = [len(c) for c in communities.values()]
        metrics['avg_community_size'] = np.mean(comm_sizes)
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {}
        
    return metrics

def create_test_graphs(n_graphs: int = 10):
    """Creates test graphs with different random seeds"""
    graphs = defaultdict(list)
    seeds = range(42, 42 + n_graphs)  
    
    for seed in seeds:
        # LFR Benchmark
        try:
            G_lfr = nx.generators.community.LFR_benchmark_graph(
                n=1000,
                tau1=2.5,
                tau2=1.5,
                mu=0.3,
                min_degree=20,
                max_degree=50,
                min_community=20,
                max_community=100,
                seed=seed
            )
            ground_truth = {node: G_lfr.nodes[node]['community'] 
                          for node in G_lfr.nodes()}
            graphs['lfr'].append((G_lfr, ground_truth))
            
        except Exception as e:
            print(f"Error generating LFR graph: {e}")
            continue
        
        # Random community graph
        try:
            n_nodes = 1000
            n_communities = 5
            G = nx.Graph()
            G.add_nodes_from(range(n_nodes))
            
            # Create ground truth
            ground_truth = {}
            nodes_per_comm = n_nodes // n_communities
            for i in range(n_nodes):
                comm = i // nodes_per_comm
                if comm < n_communities:
                    ground_truth[i] = comm
            
            # Add edges
            random.seed(seed)
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    p = 0.1 if ground_truth[i] == ground_truth[j] else 0.01
                    if random.random() < p:
                        G.add_edge(i, j)
                        
            graphs['random'].append((G, ground_truth))
            
        except Exception as e:
            print(f"Error generating random graph: {e}")
            continue
            
    return graphs

def run_algorithm_comparison(graphs: Dict, n_runs: int = 10) -> Dict:
    """Compares re_mocd algorithms"""
    results = {
        'fast_nx': CommunityMetrics(),
        'from_nx': CommunityMetrics()
    }
    
    algorithms = {
        'fast_nx': re_mocd.fast_nx,
        'from_nx': re_mocd.from_nx
    }
    
    for graph_type, graph_list in graphs.items():
        print(f"\nProcessing {graph_type} graphs...")
        
        for i, (G, ground_truth) in enumerate(graph_list):
            print(f"Graph {i+1}/{len(graph_list)}")
            
            for algo_name, algo_func in algorithms.items():
                try:
                    start_time = time.time()
                    partitions = algo_func(G, True)
                    exec_time = time.time() - start_time
                    
                    # Calculate metrics
                    metrics = validate_communities(G, partitions, ground_truth)
                    metrics['execution_time'] = exec_time
                    
                    # Store results
                    for metric, value in metrics.items():
                        results[algo_name].add_metric(
                            f"{graph_type}_{metric}", 
                            value
                        )
                        
                except Exception as e:
                    print(f"Error running {algo_name}: {e}")
                    
    return results

def print_comparison_results(results: Dict):
    """Prints statistical comparison"""
    metrics = ['modularity', 'nmi', 'execution_time', 'internal_density']
    graph_types = ['lfr', 'random']
    
    for graph_type in graph_types:
        print(f"\n{'='*60}")
        print(f"Results for {graph_type} graphs:")
        print('='*60)
        
        for metric in metrics:
            metric_key = f"{graph_type}_{metric}"
            print(f"\n{metric.upper()}:")
            
            for algo_name, algo_metrics in results.items():
                mean, ci_low, ci_high = algo_metrics.get_confidence_interval(metric_key)
                print(f"\n{algo_name}:")
                print(f"  Mean: {mean:.4f}")
                print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

def plot_comparison_results(results: Dict, save_path: str = 'comparison_results.png'):
    """Plots statistical comparison using matplotlib"""
    
    metrics = ['modularity', 'nmi', 'execution_time', 'internal_density']
    graph_types = ['lfr', 'random']
    algorithms = list(results.keys())
    
    # Setup plot grid
    fig, axes = plt.subplots(len(metrics), len(graph_types), figsize=(15, 20))
    fig.suptitle('Community Detection Algorithms Comparison', fontsize=16, y=0.95)
    
    colors = ['#2ecc71', '#3498db']    
    for i, metric in enumerate(metrics):
        for j, graph_type in enumerate(graph_types):
            ax = axes[i, j]
            metric_key = f"{graph_type}_{metric}"
            
            # Data for plotting
            x_pos = np.arange(len(algorithms))
            means = []
            errors = []
            
            for algo_name, algo_metrics in results.items():
                mean, ci_low, ci_high = algo_metrics.get_confidence_interval(metric_key)
                means.append(mean)
                errors.append([mean - ci_low, ci_high - mean])
            
            # Create bar plot
            bars = ax.bar(x_pos, means, yerr=np.array(errors).T, 
                         capsize=5, color=colors, alpha=0.8,
                         error_kw={'ecolor': 'gray', 'capthick': 2})
            
            # Customize plot
            ax.set_title(f'{graph_type.upper()} - {metric.replace("_", " ").title()}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(algorithms, rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}', ha='center', va='bottom')
            
            # Add grid for better readability
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Adjust y-axis limits to accommodate error bars
            if metric == 'execution_time':
                ax.set_ylabel('Time (seconds)')
            else:
                ax.set_ylabel('Score')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as {save_path}")

if __name__ == "__main__":
    # Configurações
    N_GRAPHS = 10
    
    print("Generating test graphs...")
    graphs = create_test_graphs(N_GRAPHS)
    
    print("\nRunning algorithm comparison...")
    results = run_algorithm_comparison(graphs)
    
    print("\nResults:")
    print_comparison_results(results)

    plot_comparison_results(results, 'community_detection_comparison.png')