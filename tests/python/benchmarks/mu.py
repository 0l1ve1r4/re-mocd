import re_mocd
import time
import random
import pandas as pd
from networkx.generators.community import LFR_benchmark_graph 
from .utils import convert_to_node_clustering, compute_nmi
from cdlib import algorithms

def make_benchmark():
    num_runs = 1
    random.seed(42)
    results = []
    mu_values = [round(0.1 * x, 1) for x in range(1, 2)]
    for mu in mu_values:
        for run in range(num_runs):
            n = 100
            min_community = max(30, n // 50)
            max_community = max(80, n // 20)
            min_degree = max(10, n // 100)
            max_degree = min(50, n // 10)

            try:
                G = LFR_benchmark_graph(n, 2.0, 3.5, mu,
                    min_degree=min_degree,
                    max_degree=max_degree,
                    min_community=min_community,
                    max_community=max_community,
                    seed=42)

                start_time = time.time()
                mocd_partition = re_mocd.from_nx(G)
                execution_time = time.time() - start_time

                mocd_nc = convert_to_node_clustering(mocd_partition, G)

                louvain_communities = algorithms.louvain(G)
                leiden_communities = algorithms.leiden(G)
                nmi_louvain = compute_nmi(mocd_partition, louvain_communities, G)
                nmi_leiden = compute_nmi(mocd_partition, leiden_communities, G)

                result = {
                    'mu': mu,
                    'run': run + 1,
                    'num_nodes': n,
                    'nmi_louvain': nmi_louvain,
                    'nmi_leiden': nmi_leiden,
                    'execution_time': execution_time,
                }
                results.append(result)

                df = pd.DataFrame(results)
                df.to_csv("benchmark_mu.csv", index=False)

            except Exception as inst:
                print(f"Error for mu={mu}, run={run+1}: {inst}")
    return results

def run(run_subprocess: bool):
    make_benchmark()