import networkx as nx
from collections import defaultdict
from cdlib import algorithms, evaluation, NodeClustering
from networkx.generators.community import LFR_benchmark_graph 
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import subprocess
import time
import random
import csv
import pandas as pd
from datetime import datetime
import os
import rmocd

def visualize_comparison(
    graph: nx.Graph, 
    partition_ga: NodeClustering, 
    partition_two: NodeClustering, 
    nmi_score: float, 
    save_file_path: str = None,
    title: str = "Algorithm"
):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # Change from spring_layout to circular_layout
    pos = nx.spring_layout(graph)

    # Visualize rmocd (partition_ga) Communities
    communities_ga = partition_ga.communities
    communities_ga_dict = defaultdict(list)
    for idx, community in enumerate(communities_ga):
        for node in community:
            communities_ga_dict[idx].append(node)
    
    # Use a different colormap that works better for circular visualization
    colors_ga = plt.cm.Set3(np.linspace(0, 1, len(communities_ga_dict)))
    color_map_ga = {
        node: color 
        for color, nodes in zip(colors_ga, communities_ga_dict.values()) 
        for node in nodes
    }
    
    # Draw first plot with improved visual parameters
    nx.draw_networkx_nodes(
        graph, 
        pos=pos, 
        nodelist=graph.nodes(),
        node_color=[color_map_ga[node] for node in graph.nodes()],
        node_size=500,  # Increase node size
        ax=axs[0]
    )
    nx.draw_networkx_edges(
        graph, 
        pos=pos, 
        ax=axs[0],
        edge_color='gray',
        width=1.0,
        alpha=0.5
    )
    nx.draw_networkx_labels(
        graph, 
        pos=pos, 
        ax=axs[0],
        font_size=8,
        font_weight='bold'
    )
    axs[0].set_title("Rmocd", pad=20)
    axs[0].axis('off')

    # Visualize the second algorithm (partition_two) Communities
    communities_algo = partition_two.communities
    communities_algo_dict = defaultdict(list)
    for idx, community in enumerate(communities_algo):
        for node in community:
            communities_algo_dict[idx].append(node)
    
    # Use the same colormap for consistency
    colors_algo = plt.cm.Set3(np.linspace(0, 1, len(communities_algo_dict)))
    color_map_algo = {
        node: color 
        for color, nodes in zip(colors_algo, communities_algo_dict.values()) 
        for node in nodes
    }
    
    # Draw second plot with improved visual parameters
    nx.draw_networkx_nodes(
        graph, 
        pos=pos, 
        nodelist=graph.nodes(),
        node_color=[color_map_algo[node] for node in graph.nodes()],
        node_size=500,  # Increase node size
        ax=axs[1]
    )
    nx.draw_networkx_edges(
        graph, 
        pos=pos, 
        ax=axs[1],
        edge_color='gray',
        width=1.0,
        alpha=0.5
    )
    nx.draw_networkx_labels(
        graph, 
        pos=pos, 
        ax=axs[1],
        font_size=8,
        font_weight='bold'
    )
    axs[1].set_title(title, pad=20)
    axs[1].axis('off')

    # Add NMI score with improved positioning
    fig.suptitle(f'NMI Score: {nmi_score:.4f}', fontsize=16, y=0.95)

    # Add padding between subplots
    plt.tight_layout(pad=3.0)

    if save_file_path is None:
        plt.show()
    else:
        plt.savefig(save_file_path, bbox_inches='tight', dpi=300)
        plt.close()

def compute_nmi(partition_ga: dict, partition_algorithm: NodeClustering, graph: nx.Graph):
    """Compute NMI between Genetic Algorithm partition (dictionary) and another partitioning algorithm."""
    communities_ga = defaultdict(list)
    for node, community in partition_ga.items():
        communities_ga[community].append(node)
    ga_communities_list = list(communities_ga.values())
    ga_node_clustering = NodeClustering(ga_communities_list, graph, "Genetic Algorithm")

    nmi_value = evaluation.normalized_mutual_information(ga_node_clustering, partition_algorithm)
    return nmi_value.score

def convert_edgelist_to_graph(edgelist_file: str):
    """Convert an edgelist to a NetworkX graph."""
    try:
        G = nx.read_edgelist(edgelist_file, delimiter=',', nodetype=int)
        return G
    except Exception as e:
        print(f"Error reading edgelist file: {e}")
        raise

def convert_to_node_clustering(partition_dict, graph):
    """Convert a dictionary partition to NodeClustering."""
    communities = defaultdict(list)
    for node, community in partition_dict.items():
        communities[community].append(node)

    community_list = list(communities.values())
    return NodeClustering(community_list, graph, "rmocd Algorithm")

def generate_ring_of_cliques(file_path: str, m: int, num_cliques: int):
    if num_cliques % 2 != 0:
        raise ValueError("Number of cliques must be even")    
    if m < 2:
        raise ValueError("Clique size must be at least 2")

    G = nx.Graph()
    for i in range(num_cliques):
        clique_nodes = range(i * m, (i + 1) * m)
        for u in clique_nodes:
            for v in clique_nodes:
                if u < v:  # Avoid duplicate edges
                    G.add_edge(u, v)        
        if i > 0:
            current_clique = list(range(i * m, (i + 1) * m))
            prev_clique = list(range((i - 1) * m, i * m))
            G.add_edge(random.choice(current_clique), random.choice(prev_clique))
    last_clique = list(range((num_cliques - 1) * m, num_cliques * m))
    first_clique = list(range(m))
    G.add_edge(random.choice(last_clique), random.choice(first_clique))
    nx.write_edgelist(G, file_path, delimiter=",", data=False)
    return G

def run_comparisons(graph_file: str, show_plot: bool):
    mocd_partition = rmocd.run(graph_file, pesa_ii=False, debug=True)
    G = convert_edgelist_to_graph(graph_file)
    mocd_nc = convert_to_node_clustering(mocd_partition, G)

    louvain_communities = algorithms.louvain(G)
    leiden_communities = algorithms.leiden(G)

    nmi_louvain = compute_nmi(mocd_partition, louvain_communities, G)
    nmi_leiden = compute_nmi(mocd_partition, leiden_communities, G)

    if show_plot:
        visualize_comparison(G, mocd_nc, louvain_communities, nmi_louvain, "output", title="louvain")
        visualize_comparison(G, mocd_nc, leiden_communities, nmi_leiden, "output", title="Leiden")


def run_mocd_subprocess(graph_file, mocd_path="./target/release/mocd", pesa_ii=False):
    """Run MOCD algorithm using subprocess to call the compiled executable."""
    try:
        cmd = [mocd_path, graph_file]
        if pesa_ii:
            cmd.append("--pesa-ii")

        process = subprocess.run(cmd,check=True)
        with open("output.json", 'r') as f:
            community_dict = json.load(f)
            return {int(node): int(community_id) for node, community_id in community_dict.items()}

    except Exception as e:
        print(f"Error running MOCD: {str(e)}")
        raise

def make_benchmark():
    # Initialize parameters
    base_n = 500  
    tau1 = 2.0
    tau2 = 3.5
    min_degree_factor = 100
    max_degree_factor = 10
    min_community_factor = 50
    max_community_factor = 20
    num_runs = 100

    random.seed(42)
    results = []

    mu_values = [round(0.1 * x, 1) for x in range(1, 10)]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"benchmark_results_{timestamp}.csv"

    for mu in mu_values:
        for run in range(num_runs):
            n = 500  
            min_community = max(30, n // min_community_factor)
            max_community = max(80, n // max_community_factor)
            min_degree = max(10, n // min_degree_factor)
            max_degree = min(50, n // max_degree_factor)

            try:
                G = LFR_benchmark_graph(
                    n,
                    tau1,
                    tau2,
                    mu,
                    min_degree=min_degree,
                    max_degree=max_degree,
                    min_community=min_community,
                    max_community=max_community,
                    seed=42 + run,  # Different seed for each run
                )

                save_path = f"temp.edgelist"
                nx.write_edgelist(
                    G,
                    save_path,
                    delimiter=",",
                )

                start_time = time.time()
                mocd_partition = run_mocd_subprocess(save_path)
                execution_time = time.time() - start_time

                mocd_nc = convert_to_node_clustering(mocd_partition, G)

                louvain_communities = algorithms.louvain(G)
                leiden_communities = algorithms.leiden(G)
                nmi_louvain = compute_nmi(mocd_partition, louvain_communities, G)
                nmi_leiden = compute_nmi(mocd_partition, leiden_communities, G)


                # Store results
                result = {
                    'mu': mu,
                    'run': run + 1,
                    'num_nodes': n,
                    'min_community': min_community,
                    'max_community': max_community,
                    'min_degree': min_degree,
                    'max_degree': max_degree,
                    'nmi_louvain': nmi_louvain,
                    'nmi_leiden': nmi_leiden,
                    'execution_time': execution_time,
                    'tau1': tau1,
                    'tau2': tau2,
                }
                results.append(result)

                # Save results after each run
                df = pd.DataFrame(results)
                df.to_csv(csv_filename, index=False)

            except Exception as inst:
                print(f"Error for mu={mu}, run={run+1}: {inst}")
                # Log the error in results
                result = {
                    'mu': mu,
                    'run': run + 1,
                    'error': str(inst),
                    'tau1': tau1,
                    'tau2': tau2,
                }
                results.append(result)

    # Calculate and save summary statistics
    summary_df = pd.DataFrame(results)
    summary_stats = summary_df.groupby('mu').agg({
        'nmi_louvain': ['mean', 'std'],
        'nmi_leiden': ['mean', 'std'],
        'execution_time': ['mean', 'std']
    }).round(4)

    summary_stats.to_csv(f'benchmark_summary_{timestamp}.csv')
    
    return results

if __name__ == "__main__":
    runs_per_file = 10
    
    generate_ring_of_cliques("ring.edgelist", 7, 6)

    has_args = (len(sys.argv) > 1)
    graph_files = None
    num_files = None
    show_plot = False

    if sys.argv[1:][0] == "--benchmark":
        make_benchmark()
        print("Done")
        exit(0)

    if has_args:
        graph_files = (sys.argv[1:])[0]
        num_files = len(sys.argv[1:])
    else: 
        graph_files = mu_graphs
        num_files = len(graph_files)

    if num_files == 1:
        show_plot = True
        run_comparisons(graph_files, show_plot)
    else:
        for i in range(num_files):
            for _ in range(runs_per_file):
                run_comparisons(graph_files[i], show_plot)

    print("Done.")