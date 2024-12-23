import networkx as nx
import os

def save_edgelist(graph_name, edgelist):
    """Saves the edge list to a file."""
    folder = "edgelists"
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{graph_name}.edgelist")
    with open(filepath, "w") as f:
        for edge in edgelist:
            f.write(f"{edge[0]},{edge[1]},{{}}\n")
    print(f"Saved {graph_name} edgelist to {filepath}")

# Define the graphs and their specifications
graphs_specs = {
    "Karate": (34, 78),
    "Lesmis": (77, 254),
    "Polbooks": (105, 441),
    "Adjnoun": (112, 425),
    "Football": (115, 613),
    "Celegansneural": (297, 2345),
    "Celegansmetabolic": (453, 2025),
    "Netscience": (1589, 2742),
    "Power": (4941, 6594),
    "Hepth": (8361, 15751)
}

for graph_name, (nodes, edges) in graphs_specs.items():
    # Generate a random graph with the specified number of nodes and edges
    graph = nx.gnm_random_graph(nodes, edges, seed=42)

    # Create the edge list
    edgelist = list(graph.edges())

    # Save the graph as an edge list in the specified format
    save_edgelist(graph_name, edgelist)
