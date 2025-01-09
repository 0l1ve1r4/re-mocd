import re_mocd
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score
from networkx.algorithms.community import modularity
from cdlib.algorithms import leiden, louvain
from cdlib import evaluation
from tabulate import tabulate  # for nice table formatting

def calculate_modularity(G, partition):
    """Calculate modularity using NetworkX's built-in function"""
    community_dict = defaultdict(set)
    for node, community_id in partition.items():
        community_dict[community_id].add(node)
    communities = list(community_dict.values())
    return modularity(G, communities)

def from_csv(file_path):
    """
    Creates a NetworkX graph from a CSV file with specified headers.
    Only includes edges from 'src' to 'trg' without weights.
    """
    df = pd.read_csv(file_path)
    G = nx.Graph()
    G.add_edges_from(df[['src', 'trg']].values)
    return G

def get_community_count(partition):
    """Count number of unique communities"""
    return len(set(partition.values()))

def calculate_nmi(partition1, partition2):
    """Calculate NMI between two partitions"""
    # Convert partitions to lists of community labels
    nodes = list(partition1.keys())
    labels1 = [partition1[node] for node in nodes]
    labels2 = [partition2[node] for node in nodes]
    return normalized_mutual_info_score(labels1, labels2)

def modularity_analysis():
    # Load your graph
    G = from_csv("tests/python/RIOTS-edgelist.csv")
    print(f"Graph Info - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}\n")

    # Run RMOCd
    rmocd_partition = re_mocd.fast(G, True)
    rmocd_modularity = calculate_modularity(G, rmocd_partition)
    rmocd_communities = get_community_count(rmocd_partition)

    # Run Leiden
    leiden_result = leiden(G)
    leiden_partition = {node: i for i, community in enumerate(leiden_result.communities) 
                       for node in community}
    leiden_modularity = calculate_modularity(G, leiden_partition)
    leiden_communities = get_community_count(leiden_partition)

    # Run Louvain
    louvain_result = louvain(G)
    louvain_partition = {node: i for i, community in enumerate(louvain_result.communities) 
                        for node in community}
    louvain_modularity = calculate_modularity(G, louvain_partition)
    louvain_communities = get_community_count(louvain_partition)

    # Calculate NMI between all pairs
    nmi_rmocd_leiden = calculate_nmi(rmocd_partition, leiden_partition)
    nmi_rmocd_louvain = calculate_nmi(rmocd_partition, louvain_partition)
    nmi_leiden_louvain = calculate_nmi(leiden_partition, louvain_partition)

    # Create comparison table
    headers = ["Algorithm", "Modularity", "Communities", "NMI (RMOCd)", "NMI (Leiden)", "NMI (Louvain)"]
    table = [
        ["RMOCd", f"{rmocd_modularity:.4f}", rmocd_communities, "1.000", 
         f"{nmi_rmocd_leiden:.4f}", f"{nmi_rmocd_louvain:.4f}"],
        ["Leiden", f"{leiden_modularity:.4f}", leiden_communities, 
         f"{nmi_rmocd_leiden:.4f}", "1.000", f"{nmi_leiden_louvain:.4f}"],
        ["Louvain", f"{louvain_modularity:.4f}", louvain_communities, 
         f"{nmi_rmocd_louvain:.4f}", f"{nmi_leiden_louvain:.4f}", "1.000"]
    ]

    print("\nCommunity Detection Comparison:")
    print(tabulate(table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    modularity_analysis()