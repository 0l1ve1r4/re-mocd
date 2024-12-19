import networkx as nx
import random 

from networkx.generators.community import LFR_benchmark_graph

def article_example_graph():
    G = nx.Graph()
    edges = [
            (0, 1), (0, 2), (0, 3), (0, 7), 
            (1, 2), (1, 3), (1, 5), (1, 7),
            (2, 3),
            (4, 5), 
            (5, 6), 
            (6, 4), 
            (6, 8), 
            (7, 8)
            ]

    G.add_edges_from(edges)
    return G

def benchmark_graph():
    # LFR
    n = 400         # Nodes num
    tau1 = 2.0      # Expoente do grau
    tau2 = 3.5      # Expoente do tamanho da comunidade
    mu = 0.05        # Taxa de mistura
    min_community = 40
    max_community = 50

    # Gerando o grafo LFR
    random.seed(42)
    G = LFR_benchmark_graph(
    n, tau1, tau2, mu, min_degree=10, max_degree=50,
    min_community=min_community, max_community=max_community, seed=42
    )

    return G