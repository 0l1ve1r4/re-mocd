import utils
import random
import plots as pl
import networkx as nx
import genetic_algorithm as ga

from collections import defaultdict
from cdlib import algorithms, evaluation, NodeClustering

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


if __name__ == "__main__":
    G = nx.karate_club_graph()
    G = utils.benchmark_graph()

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
    pl.visualize_comparison(G, best_partition, louvain_communities, nmi_score)

    # =============================================================================================
    # Extras Visuzalitions
    # ==============================================================================================
    # exit(0)

    pl.plot_fitness_history(best_history, avg_history)
    pl.visualize_all(G, best_partition)
    best_fitness = ga.calculate_objectives(G, best_partition)

