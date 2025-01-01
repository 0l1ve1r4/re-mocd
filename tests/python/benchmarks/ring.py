from .utils import run_comparisons
import networkx as nx
import random

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
    return G

def run():
    run_comparisons(generate_ring_of_cliques("ring.edgelist", 10, 8), 
                    True)