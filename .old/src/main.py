import utils
import random
import plots as pl
import pandas as pd
import networkx as nx
import genetic_algorithm as ga

"""
1. 10 runs variar o mu 0.1 - 0.9
    - calcular a media do nmi par a par das 10 execuções para cada algoritmo
    - plot grafico de linha com intervalo de confiança y - nmi media, x - mu
    
aumentar mu 

"""

from collections import defaultdict
from cdlib import algorithms, evaluation, NodeClustering
from networkx.generators.community import LFR_benchmark_graph

NUM_GENERATIONS = 400
POPULATION_SIZE = 100


def compute_nmi(partition_ga, partition_louvain, graph):
    # Convert GA partition to CDLIB NodeClustering format
    communities_ga = defaultdict(list)
    for node, community in partition_ga.items():
        communities_ga[community].append(node)
    ga_communities_list = [community for community in communities_ga.values()]
    ga_node_clustering = NodeClustering(ga_communities_list, graph, "Genetic Algorithm")

    nmi_value = evaluation.normalized_mutual_information(
        ga_node_clustering, partition_louvain
    )
    return nmi_value.score


data = []

def convert_edgelist_to_graph(edgelist_file):
    """Convert an edgelist to a NetworkX graph."""
    G = nx.read_edgelist(edgelist_file, delimiter=',', nodetype=int)
    return G


if __name__ == "__main__":
    for i in range(1, 2):
        try:
            try:
                G = convert_edgelist_to_graph("/home/ol1ve1r4/Desktop/mocd/src/graphs/artificials/mu-0.1.edgelist")
                
            except Exception as e:
                print(f"Failed to generate graph at iteration {i}: {e}")
                continue

            # Louvain Algorithm
            louvain_communities = algorithms.louvain(G)

            # Ga Visualization - Run the genetic algorithm with Max-Min Distance selection
            (
                best_partition,
                deviations,
                real_fitnesses,
                random_fitnesses,
                best_history,
                avg_history,
            ) = ga.genetic_algorithm(G, NUM_GENERATIONS, POPULATION_SIZE)

            # Visualize GA x Louvain
            nmi_score = compute_nmi(best_partition, louvain_communities, G)
            pl.visualize_comparison(
                G, best_partition, louvain_communities, nmi_score, f"gen_{i}"
            )

            # Save information in the DataFrame
            for generation in range(NUM_GENERATIONS):
                data.append(
                    {
                        "generation": generation,
                        "best_history": best_history[generation],
                        "avg_history": avg_history[generation],
                    }
                )
        except KeyboardInterrupt:
            break

    df = pd.DataFrame(data)
    df.to_csv("generations_data.csv", index=False)
    print("DataFrame saved to generations_data.csv")

    # =============================================================================================
    # Extras Visuzalitions
    # ==============================================================================================
    exit(0)

    pl.plot_fitness_history(best_history, avg_history)
    pl.visualize_all(G, best_partition)
    best_fitness = ga.calculate_objectives(G, best_partition)
