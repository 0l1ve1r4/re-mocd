import networkx as nx
from collections import defaultdict
from cdlib import algorithms, evaluation, NodeClustering
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import subprocess

def visualize_comparison(graph: nx.Graph, partition_ga: dict, partition_two: NodeClustering, nmi_score: float, save_file_path: str = None):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    pos = nx.spring_layout(graph, seed=42)

    communities_ga = defaultdict(list)
    for node, community in partition_ga.items():
        communities_ga[community].append(node)
    colors_ga = plt.cm.rainbow(np.linspace(0, 1, len(communities_ga)))
    color_map_ga = {node: color for color, nodes in zip(colors_ga, communities_ga.values()) for node in nodes}
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=graph.nodes(),
                           node_color=[color_map_ga[node] for node in graph.nodes()], ax=axs[0])
    nx.draw_networkx_edges(graph, pos=pos, ax=axs[0])
    nx.draw_networkx_labels(graph, pos=pos, ax=axs[0])  # Add node labels (numbers)
    axs[0].set_title("MOCD - GA/Pareto")
    axs[0].axis('off')

    # Second Algorithm (Louvain/Leiden) Communities Visualization
    communities_algo = {node: idx for idx, community in enumerate(partition_two.communities) for node in community}
    communities_algo_dict = defaultdict(list)
    for node, community in communities_algo.items():
        communities_algo_dict[community].append(node)
    colors_algo = plt.cm.rainbow(np.linspace(0, 1, len(communities_algo_dict)))
    color_map_algo = {node: color for color, nodes in zip(colors_algo, communities_algo_dict.values()) for node in nodes}
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=graph.nodes(),
                           node_color=[color_map_algo[node] for node in graph.nodes()], ax=axs[1])
    nx.draw_networkx_edges(graph, pos=pos, ax=axs[1])
    nx.draw_networkx_labels(graph, pos=pos, ax=axs[1])  # Add node labels (numbers)
    axs[1].set_title("Second Algorithm (Louvain/Leiden)")
    axs[1].axis('off')

    fig.suptitle(f'NMI Score: {nmi_score:.4f}', fontsize=16)

    if save_file_path is None:
        plt.show()
    else:
        plt.savefig(save_file_path)

def compute_nmi(partition_ga: dict, partition_algorithm: NodeClustering, graph: nx.Graph):
    """Compute NMI between Genetic Algorithm and another partitioning algorithm."""
    communities_ga = defaultdict(list)
    for node, community in partition_ga.items():
        communities_ga[community].append(node)
    ga_communities_list = list(communities_ga.values())
    ga_node_clustering = NodeClustering(ga_communities_list, graph, "Genetic Algorithm")

    nmi_value = evaluation.normalized_mutual_information(
        ga_node_clustering, partition_algorithm
    )
    return nmi_value.score

def convert_edgelist_to_graph(edgelist_file: str):
    """Convert an edgelist to a NetworkX graph."""
    try:
        G = nx.read_edgelist(edgelist_file, delimiter=',', nodetype=int)
        return G
    except Exception as e:
        print(f"Error reading edgelist file: {e}")
        raise

def load_json_partition(file_path):
    """Load the best partition from a JSON file output by the Rust algorithm."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {int(node): int(community) for node, community in data.items()}


def mocd(graph_file: str, mocd_executable: str):
    args = [graph_file]
    try:
        result = subprocess.run(
            [rust_executable, *args],   
            text=True,                
            capture_output=True,       
            check=True                 
        )

    except subprocess.CalledProcessError as e:
        print(f"Error:\n{e.stderr}")
    except FileNotFoundError:
        print(f"MOCD not found: {rust_executable}")

def run_comparisons(graph_file: str, mocd_json:str, rust_executable:str, csv_file:str):
    # Run the rust algorithm
    mocd(graph_file, rust_executable)
    mocd_partition = load_json_partition(mocd_json)

    # Run Louvain and Leiden Algorithms
    G = convert_edgelist_to_graph(graph_file)
    louvain_communities = algorithms.louvain(G)
    leiden_communities = algorithms.leiden(G)
    nmi_louvain = compute_nmi(mocd_partition, louvain_communities, G)
    nmi_leiden = compute_nmi(mocd_partition, leiden_communities, G)

    # Save output
    with open(csv_file, "r+") as file:
        lines = file.readlines()
        if lines:
            lines[-1] = lines[-1].strip() + f",{nmi_louvain},{nmi_leiden}\n"
        else:
            lines.append("elapsed_time,num_nodes,num_edges,modularity,nmi_louvain,nmi_leiden\n")
            lines.append(f"0,0,0,0,{nmi_louvain:.4},{nmi_leiden:.4}\n")
        file.seek(0)
        file.writelines(lines)

    visualize_comparison(G, mocd_partition, louvain_communities, nmi_louvain)
    visualize_comparison(G, mocd_partition, leiden_communities, nmi_leiden)

if __name__ == "__main__":
    mocd_json = "res/output.json"
    rust_executable = "./target/release/mocd"
    csv_file = "res/mocd_output.csv"
    runs_per_file = 10
    
    graph_file = (sys.argv[1:])[0]
    num_files = len(sys.argv) - 1

    if num_files == 1:
        run_comparisons(graph_file, mocd_json, rust_executable, csv_file)
    
    else:
        for i in range(runs_per_file):
            run_comparisons(graph_file[i], mocd_json, rust_executable, csv_file)

    print("\nDone.")
