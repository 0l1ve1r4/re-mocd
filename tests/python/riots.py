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
from tabulate import tabulate 

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
    nx.write_edgelist(G, "rmocd.edgelist", delimiter=',')
    print(f"Graph Info - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}\n")
    
    num_runs = 1
    max_runs = 1
    
    for run in range(1, num_runs + 1):
        print(f"Run {run} of {num_runs}")
        
        best_rmocd_modularity = 0
        best_leiden_modularity = 0
        best_louvain_modularity = 0
        best_rmocd_values = {}
        best_leiden_values = {}
        best_louvain_values = {}
        
        for i in range(max_runs):
            # Run RMOCd
            rmocd_partition = re_mocd.from_nx(G, multi_level=True, debug=True)
            rmocd_modularity = calculate_modularity(G, rmocd_partition)
            rmocd_communities = get_community_count(rmocd_partition)

            if rmocd_modularity > best_rmocd_modularity:
                best_rmocd_modularity = rmocd_modularity
                best_rmocd_values = {
                    "modularity": rmocd_modularity,
                    "communities": rmocd_communities
                }

            # Run Leiden
            leiden_result = leiden(G)
            leiden_partition = {node: i for i, community in enumerate(leiden_result.communities) 
                                for node in community}
            leiden_modularity = calculate_modularity(G, leiden_partition)
            leiden_communities = get_community_count(leiden_partition)

            if leiden_modularity > best_leiden_modularity:
                best_leiden_modularity = leiden_modularity
                best_leiden_values = {
                    "modularity": leiden_modularity,
                    "communities": leiden_communities
                }

            # Run Louvain
            louvain_result = louvain(G)
            louvain_partition = {node: i for i, community in enumerate(louvain_result.communities) 
                                for node in community}
            louvain_modularity = calculate_modularity(G, louvain_partition)
            louvain_communities = get_community_count(louvain_partition)

            if louvain_modularity > best_louvain_modularity:
                best_louvain_modularity = louvain_modularity
                best_louvain_values = {
                    "modularity": louvain_modularity,
                    "communities": louvain_communities
                }
        
        # Print the best results for this run
        print(f"Best Modularity and Communities for Run {run}:")
        print(f"RMOCd: Modularity = {best_rmocd_values['modularity']:.4f}, "
              f"Communities = {best_rmocd_values['communities']}")
        print(f"Leiden: Modularity = {best_leiden_values['modularity']:.4f}, "
              f"Communities = {best_leiden_values['communities']}")
        print(f"Louvain: Modularity = {best_louvain_values['modularity']:.4f}, "
              f"Communities = {best_louvain_values['communities']}")
        print("-" * 50)


if __name__ == "__main__":
    modularity_analysis()