"""
// This Source Code Form is subject to the terms of The GNU General Public License v3.0
// Copyright 2024 - Guilherme Santos. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://www.gnu.org/licenses/gpl-3.0.html
"""

import random   
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph as benchmark

if __name__ == "__main__":
    n = 4000 
    tau1 = 2.0
    tau2 = 3.5
    mu = 0.4
    min_community = max(10, n // 50)
    max_community = max(20, n // 20)
    min_degree = max(10, n // 100)
    max_degree = min(50, n // 10)

    random.seed(42)
    try:
        G = benchmark (
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
        
        save_path = (f"../graphs/artificials/mu-{mu}.edgelist") 
        nx.write_edgelist(
            G,
            save_path,
            delimiter=",",
        )
        
    except Exception as inst:
        print(inst)
