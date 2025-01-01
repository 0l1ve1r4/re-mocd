import networkx as nx
from .utils import run_comparisons

def run(file_path: str):
    with open(file_path, 'r') as f:
        edges = []
        for line in f:
            parts = line.split(",")
            u, v = int(parts[0]), int(parts[1])  # Parse node ids
            edges.append((u, v))  # Ignore `{}`

    G = nx.from_edgelist(edges)
    run_comparisons(G, True)
