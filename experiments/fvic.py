import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import re_mocd
from collections import defaultdict
from community import community_louvain
import leidenalg
import igraph as ig

def convert_to_igraph(G):
    edges = list(G.edges())
    return ig.Graph(n=G.number_of_nodes(), edges=edges)

def leiden_to_dict(partition):
    return {node: comm for node, comm in enumerate(partition.membership)}

def louvain_to_dict(partition):
    return {node: int(comm) for node, comm in partition.items()}

def generate_symmetric_network(n_communities=4, size_community=32, zout=2, seed=42):
    """Gera rede simétrica onde todas as comunidades têm o mesmo tamanho"""
    np.random.seed(seed)
    total_nodes = n_communities * size_community
    p_in = (16 - zout) / (size_community - 1)
    p_out = zout / (total_nodes - size_community)
    
    G = nx.Graph()
    G.add_nodes_from(range(total_nodes))
    ground_truth = {}
    
    for comm in range(n_communities):
        nodes = range(comm * size_community, (comm + 1) * size_community)
        for i in nodes:
            ground_truth[i] = comm
            for j in nodes:
                if i < j and np.random.random() < p_in:
                    G.add_edge(i, j)
    
    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            if ground_truth[i] != ground_truth[j]:
                if np.random.random() < p_out:
                    G.add_edge(i, j)
    
    return G, ground_truth

def generate_node_asymmetric_network(zout=2, seed=42):
    """Gera rede com comunidades de tamanhos diferentes"""
    np.random.seed(seed)
    community_sizes = [96, 32]  # Uma comunidade grande e uma pequena
    total_nodes = sum(community_sizes)
    
    G = nx.Graph()
    G.add_nodes_from(range(total_nodes))
    ground_truth = {}
    
    start_idx = 0
    for comm_id, size in enumerate(community_sizes):
        p_in = (16 - zout) / (size - 1)
        nodes = range(start_idx, start_idx + size)
        
        for i in nodes:
            ground_truth[i] = comm_id
            for j in nodes:
                if i < j and np.random.random() < p_in:
                    G.add_edge(i, j)
        
        start_idx += size
    
    # Adiciona arestas entre comunidades
    p_out = zout / (total_nodes - min(community_sizes))
    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            if ground_truth[i] != ground_truth[j]:
                if np.random.random() < p_out:
                    G.add_edge(i, j)
    
    return G, ground_truth

def generate_edge_asymmetric_network(zout=2, seed=42):
    """Gera rede com diferentes densidades de conexões entre comunidades"""
    np.random.seed(seed)
    n_nodes = 128
    size_community = 64  # Duas comunidades de mesmo tamanho
    
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    ground_truth = {}
    
    # Primeira comunidade com grau médio 8
    p_in_1 = 8 / (size_community - 1)
    # Segunda comunidade com grau médio 24
    p_in_2 = 24 / (size_community - 1)
    
    # Adiciona arestas internas às comunidades
    for i in range(size_community):
        ground_truth[i] = 0
        for j in range(i + 1, size_community):
            if np.random.random() < p_in_1:
                G.add_edge(i, j)
    
    for i in range(size_community, n_nodes):
        ground_truth[i] = 1
        for j in range(i + 1, n_nodes):
            if np.random.random() < p_in_2:
                G.add_edge(i, j)
    
    # Adiciona arestas entre comunidades
    p_out = zout / size_community
    for i in range(size_community):
        for j in range(size_community, n_nodes):
            if np.random.random() < p_out:
                G.add_edge(i, j)
    
    return G, ground_truth

def calculate_fvic(detected_communities, ground_truth):
    """Calcula FVIC entre comunidades detectadas e ground truth"""
    n_nodes = len(ground_truth)
    detected_groups = defaultdict(set)
    true_groups = defaultdict(set)
    
    for node, comm in detected_communities.items():
        detected_groups[comm].add(node)
    for node, comm in ground_truth.items():
        true_groups[comm].add(node)
    
    correct_nodes = 0
    for detected_comm in detected_groups.values():
        max_overlap = 0
        for true_comm in true_groups.values():
            overlap = len(detected_comm.intersection(true_comm))
            max_overlap = max(max_overlap, overlap)
        correct_nodes += max_overlap
    
    return correct_nodes / n_nodes

def count_communities(communities):
    """Conta o número de comunidades únicas"""
    return len(set(communities.values()))

def run_algorithms(G, ground_truth):
    """Executa todos os algoritmos e retorna FVIC e número de comunidades"""
    results = {'fvic': {}, 'n_communities': {}}
    
    try:
        communities_fast = re_mocd.rmocd(G)
        results['fvic']['re-mocd'] = calculate_fvic(communities_fast, ground_truth)
        results['n_communities']['re-mocd'] = count_communities(communities_fast)
    except Exception as e:
        print(f"Erro re-mocd: {e}")
        results['fvic']['re-mocd'] = 0
        results['n_communities']['re-mocd'] = 0
    
    try:
        communities_from_nx = re_mocd.mocd(G)
        results['fvic']['mocd'] = calculate_fvic(communities_from_nx, ground_truth)
        results['n_communities']['mocd'] = count_communities(communities_from_nx)
    except Exception as e:
        print(f"Erro mocd: {e}")
        results['fvic']['mocd'] = 0
        results['n_communities']['mocd'] = 0
    
    try:
        communities_louvain = louvain_to_dict(community_louvain.best_partition(G))
        results['fvic']['Louvain'] = calculate_fvic(communities_louvain, ground_truth)
        results['n_communities']['Louvain'] = count_communities(communities_louvain)
    except Exception as e:
        print(f"Erro Louvain: {e}")
        results['fvic']['Louvain'] = 0
        results['n_communities']['Louvain'] = 0
    
    try:
        G_ig = convert_to_igraph(G)
        partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
        communities_leiden = leiden_to_dict(partition)
        results['fvic']['Leiden'] = calculate_fvic(communities_leiden, ground_truth)
        results['n_communities']['Leiden'] = count_communities(communities_leiden)
    except Exception as e:
        print(f"Erro Leiden: {e}")
        results['fvic']['Leiden'] = 0
        results['n_communities']['Leiden'] = 0
    
    return results

# Configurações do experimento
zout_values = range(0, 9, 1)
n_runs = 10
algorithms = ['re-mocd', 'mocd', 'Louvain', 'Leiden']
network_types = ['Symmetric', 'Node Asymmetric', 'Edge Asymmetric']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

# Dicionário para armazenar resultados
results = {
    net_type: {
        metric: {
            alg: {'mean': [], 'std': []} for alg in algorithms
        } for metric in ['fvic', 'n_communities']
    } for net_type in network_types
}

# Executa experimentos para cada tipo de rede
for zout in zout_values:
    print(f"Processando Zout = {zout}")
    
    for net_type in network_types:
        runs_results = {
            metric: {alg: [] for alg in algorithms} 
            for metric in ['fvic', 'n_communities']
        }
        
        for _ in range(n_runs):
            if net_type == 'Symmetric':
                G, ground_truth = generate_symmetric_network(zout=zout)
            elif net_type == 'Node Asymmetric':
                G, ground_truth = generate_node_asymmetric_network(zout=zout)
            else:  # Edge Asymmetric
                G, ground_truth = generate_edge_asymmetric_network(zout=zout)
            
            run_results = run_algorithms(G, ground_truth)
            
            for metric in ['fvic', 'n_communities']:
                for alg in algorithms:
                    runs_results[metric][alg].append(run_results[metric][alg])
        
        # Calcula estatísticas
        for metric in ['fvic', 'n_communities']:
            for alg in algorithms:
                results[net_type][metric][alg]['mean'].append(np.mean(runs_results[metric][alg]))
                results[net_type][metric][alg]['std'].append(np.std(runs_results[metric][alg]))

# Plotagem
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for col, net_type in enumerate(network_types):
    # Plot FVIC (primeira linha)
    ax_fvic = axes[0, col]
    for alg, color, marker in zip(algorithms, colors, markers):
        ax_fvic.errorbar(zout_values, 
                        results[net_type]['fvic'][alg]['mean'],
                        yerr=results[net_type]['fvic'][alg]['std'],
                        label=alg,
                        color=color,
                        marker=marker,
                        capsize=3,
                        markersize=6)
    
    ax_fvic.set_xlabel('Zout')
    ax_fvic.set_ylabel('FVIC' if col == 0 else '')
    # ax_fvic.set_title(f'Rede {net_type} - FVIC')
    ax_fvic.grid(True, linestyle='--', alpha=0.7)
    ax_fvic.set_ylim(-0.05, 1.05)
    
    # Plot número de comunidades (segunda linha)
    ax_nc = axes[1, col]
    for alg, color, marker in zip(algorithms, colors, markers):
        ax_nc.errorbar(zout_values, 
                      results[net_type]['n_communities'][alg]['mean'],
                      yerr=results[net_type]['n_communities'][alg]['std'],
                      label=alg,
                      color=color,
                      marker=marker,
                      capsize=3,
                      markersize=6)
    
    ax_nc.set_xlabel('Zout')