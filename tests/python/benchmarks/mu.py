import re_mocd
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.generators.community import LFR_benchmark_graph 
from .utils import convert_to_node_clustering, compute_nmi
from cdlib import algorithms, evaluation, NodeClustering

def make_benchmark():
    num_runs = 10
    random.seed(42)
    results = []
    mu_values = [round(0.1 * x, 1) for x in range(1, 11)]
    
    for mu in mu_values:
        for run in range(num_runs):
            n = 500
            min_community = max(30, n // 50)
            max_community = max(80, n // 20)
            min_degree = max(10, n // 100)
            max_degree = min(50, n // 10)

            try:
                # Generate the benchmark graph
                G = LFR_benchmark_graph(n, 2.0, 3.5, mu,
                    min_degree=min_degree,
                    max_degree=max_degree,
                    min_community=min_community,
                    max_community=max_community,
                    seed=42)

                # Get ground truth communities
                ground_truth = {node: G.nodes[node]['community'] for node in G.nodes()}

                # Run MOCD
                start_time = time.time()
                mocd_partition = re_mocd.from_nx(G, multi_level=False, debug=False)
                mocd_time = time.time() - start_time

                # Convert partitions to cdlib format for evaluation
                mocd_nc = convert_to_node_clustering(mocd_partition, G)
                ground_truth_nc = NodeClustering(
                    [list(nodes) for nodes in set(map(tuple, ground_truth.values()))],
                    G
                )

                # Run other algorithms
                start_time = time.time()
                louvain_communities = algorithms.louvain(G)
                louvain_time = time.time() - start_time

                start_time = time.time()
                leiden_communities = algorithms.leiden(G)
                leiden_time = time.time() - start_time

                nmi_mocd = evaluation.normalized_mutual_information(ground_truth_nc, mocd_nc).score
                nmi_louvain = evaluation.normalized_mutual_information(ground_truth_nc, louvain_communities).score
                nmi_leiden = evaluation.normalized_mutual_information(ground_truth_nc, leiden_communities).score

                # Calculate modularity
                mod_mocd = evaluation.newman_girvan_modularity(G, mocd_nc)[2]
                mod_louvain = evaluation.newman_girvan_modularity(G, louvain_communities)[2]
                mod_leiden = evaluation.newman_girvan_modularity(G, leiden_communities)[2]

                result = {
                    'mu': mu,
                    'run': run + 1,
                    'num_nodes': n,
                    'nmi_mocd': nmi_mocd,
                    'nmi_louvain': nmi_louvain,
                    'nmi_leiden': nmi_leiden,
                    'modularity_mocd': mod_mocd,
                    'modularity_louvain': mod_louvain,
                    'modularity_leiden': mod_leiden,
                    'time_mocd': mocd_time,
                    'time_louvain': louvain_time,
                    'time_leiden': leiden_time
                }
                results.append(result)

            except Exception as inst:
                print(f"Error for mu={mu}, run={run+1}: {inst}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    
    # Plot results
    plot_results(df)
    
    return results

def plot_results(df):
    # Set the style for better-looking plots
    plt.style.use('seaborn-v0_8')
    
    # Plot NMI comparison with confidence intervals
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='mu', y='nmi_mocd', label='MOCD', marker='o', ci=95)
    sns.lineplot(data=df, x='mu', y='nmi_louvain', label='Louvain', marker='s', ci=95)
    sns.lineplot(data=df, x='mu', y='nmi_leiden', label='Leiden', marker='^', ci=95)
    plt.xlabel('Mixing Parameter (μ)')
    plt.ylabel('NMI with Ground Truth')
    plt.title('NMI Comparison with Ground Truth (95% CI)')
    plt.grid(True)
    plt.savefig('nmi_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot modularity comparison with confidence intervals
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='mu', y='modularity_mocd', label='MOCD', marker='o', ci=95)
    sns.lineplot(data=df, x='mu', y='modularity_louvain', label='Louvain', marker='s', ci=95)
    sns.lineplot(data=df, x='mu', y='modularity_leiden', label='Leiden', marker='^', ci=95)
    plt.xlabel('Mixing Parameter (μ)')
    plt.ylabel('Modularity')
    plt.title('Modularity Comparison (95% CI)')
    plt.grid(True)
    plt.savefig('modularity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot execution time comparison with confidence intervals
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='mu', y='time_mocd', label='MOCD', marker='o', ci=95)
    sns.lineplot(data=df, x='mu', y='time_louvain', label='Louvain', marker='s', ci=95)
    sns.lineplot(data=df, x='mu', y='time_leiden', label='Leiden', marker='^', ci=95)
    plt.xlabel('Mixing Parameter (μ)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison (95% CI)')
    plt.grid(True)
    plt.savefig('execution_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def run(run_subprocess: bool):
    make_benchmark()