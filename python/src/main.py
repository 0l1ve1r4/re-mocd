import utils
import random
import plots as pl
import pandas as pd
import networkx as nx
import genetic_algorithm as ga

from collections import defaultdict
from cdlib import algorithms, evaluation, NodeClustering
from networkx.generators.community import LFR_benchmark_graph

NUM_GENERATIONS = 800
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

if __name__ == "__main__":
    for i in range(1, 2):
        try:
            n = 100 * i
            tau1 = 2.0
            tau2 = 3.5
            mu = min(0.05 * i, 0.7)
            min_community = max(10, n // 50)
            max_community = max(20, n // 20)
            min_degree = max(10, n // 100)
            max_degree = min(50, n // 10)

            random.seed(42)
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
                    seed=42,
                )
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
                        "iteration": i,
                        "nodes_num": n,
                        "tau1": tau1,
                        "tau2": tau2,
                        "mu": mu,
                        "min_community": min_community,
                        "max_community": max_community,
                        "generation": generation,
                        "best_history": best_history[generation],
                        "avg_history": avg_history[generation],
                        "nmi_score": nmi_score,
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

