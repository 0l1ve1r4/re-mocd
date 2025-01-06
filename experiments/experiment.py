import networkx as nx
import matplotlib.pyplot as plt
from re_mocd import re_mocd, mocd, modularity, extended_detection, default_detection
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
import networkx as nx

def convert_nodes_to_int(G):
    mapping = {node: i for i, node in enumerate(G.nodes())}
    return nx.relabel_nodes(G, mapping), mapping

def load_network(name):
    G = None
    true_communities = None
    
    if name == "karate":
        G = nx.karate_club_graph()
        true_communities = {node: 0 if node < 17 else 1 for node in G.nodes()}
        
    elif name == "davis":
        G = nx.davis_southern_women_graph()
        G, mapping = convert_nodes_to_int(G)
        women_names = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]
        true_communities = {node: 0 if node in women_names else 1 for node in G.nodes()}
        
    elif name == "florentine":
        G = nx.florentine_families_graph()
        medici_faction = {'Medici', 'Barbadori', 'Ridolfi', 'Tornabuoni', 'Albizzi', 'Salviati'}
        G, mapping = convert_nodes_to_int(G)
        medici_faction_int = {mapping[name] for name in medici_faction if name in mapping}
        true_communities = {node: 0 if node in medici_faction_int else 1 for node in G.nodes()}
        
    elif name == "les_miserables":
        G = nx.les_miserables_graph()
        G, _ = convert_nodes_to_int(G)
        true_communities = None
    
    else:
        raise ValueError(f"Network '{name}' not recognized. Available options are: "
                       "'karate', 'davis', 'florentine', 'les_miserables', 'dolphins'")
    
    return G, true_communities

def run_experiment(network_name, num_runs=10):
    G, true_communities = load_network(network_name)
    
    results = {
        'remocd': {'modularity': [], 'nmi': [], 'best_mod_comm': None, 'best_nmi_comm': None},
        'mocd': {'modularity': [], 'nmi': [], 'best_mod_comm': None, 'best_nmi_comm': None},
    }
    
    # Track best results
    best_scores = {algo: {'mod': -1, 'nmi': -1} for algo in results.keys()}
    
    for _ in range(num_runs):
        for algo_name, algo_func in [('remocd', re_mocd), ('mocd', mocd)]:
            communities = algo_func(G)
            mod = modularity(G, communities)
            nmi = normalized_mutual_info_score(
                [true_communities[n] for n in G.nodes()],
                [communities[n] for n in G.nodes()]
            )
            
            # Update best scores and communities
            if mod > best_scores[algo_name]['mod']:
                best_scores[algo_name]['mod'] = mod
                results[algo_name]['best_mod_comm'] = communities
            if nmi > best_scores[algo_name]['nmi']:
                best_scores[algo_name]['nmi'] = nmi
                results[algo_name]['best_nmi_comm'] = communities

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    pos = nx.spring_layout(G)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # Left plot (Highest NMI)
    for idx, (algo_name, color) in enumerate(zip(results.keys(), colors)):
        communities = results[algo_name]['best_nmi_comm']
        nx.draw_networkx_nodes(G, pos,
                             node_color=[communities[n] for n in G.nodes()],
                             node_size=200,
                             cmap=plt.cm.tab20,
                             label=f"{algo_name.upper()}\nNMI={best_scores[algo_name]['nmi']:.4f}",
                             ax=ax1)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax1)
    ax1.set_title(f'Highest NMI Communities\n{network_name}')
    ax1.legend()
    
    # Right plot (Highest Modularity)
    for idx, (algo_name, color) in enumerate(zip(results.keys(), colors)):
        communities = results[algo_name]['best_mod_comm']
        nx.draw_networkx_nodes(G, pos,
                             node_color=[communities[n] for n in G.nodes()],
                             node_size=200,
                             cmap=plt.cm.tab20,
                             label=f"{algo_name.upper()}\nMod={best_scores[algo_name]['mod']:.4f}",
                             ax=ax2)
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax2)
    ax2.set_title(f'Highest Modularity Communities\n{network_name}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{network_name}_results.png', dpi=300, bbox_inches='tight')
    plt.close()

# Run experiments
networks = ["davis", "florentine", "karate"]
for network in networks:
    run_experiment(network)