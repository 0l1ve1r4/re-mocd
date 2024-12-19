import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import defaultdict
import math

import random


def visualize_all(graph, partition):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    pos = nx.spring_layout(graph)
    
    # Original Graph
    axs[0].set_title("Original Graph")
    nx.draw_networkx(graph, pos=pos, ax=axs[0], with_labels=True)
    
    # Graph with Communities
    communities = defaultdict(list)
    for node, community in partition.items():
        communities[community].append(node)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))
    color_map = {}
    for color, nodes in zip(colors, communities.values()):
        for node in nodes:
            color_map[node] = color
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=graph.nodes(), node_color=[color_map[node] for node in graph.nodes()], ax=axs[1])
    nx.draw_networkx_edges(graph, pos=pos, ax=axs[1])
    axs[1].set_title("Graph with Communities")
    
    # Graph Showing Only Communities with Colorized Nodes
    new_graph = nx.Graph()
    for community_nodes in communities.values():
        new_graph.add_nodes_from(community_nodes)
        subgraph = graph.subgraph(community_nodes)
        new_graph.add_edges_from(subgraph.edges())
    # Colorize nodes in the new graph
    nx.draw_networkx_nodes(new_graph, pos=pos, nodelist=new_graph.nodes(), node_color=[color_map[node] for node in new_graph.nodes()], ax=axs[2])
    nx.draw_networkx_edges(new_graph, pos=pos, ax=axs[2])
    nx.draw_networkx_labels(new_graph, pos=pos, labels={node: node for node in new_graph.nodes()}, ax=axs[2])
    axs[2].set_title("Graph Showing Only Communities")
    
    plt.show()

# Plot Intra vs. Inter Values with Max-Min Distance
def plot_intra_inter(deviations, real_fitnesses, random_fitnesses, best_fitness):
    plt.figure(figsize=(8, 6))

    # Plot real Pareto front
    real_intra = [fitness[1] for fitness in real_fitnesses]
    real_inter = [fitness[2] for fitness in real_fitnesses]
    plt.scatter(real_intra, real_inter, label='Real Pareto Front', color='blue')

    # Plot random Pareto front
    random_intra = [fitness[1] for fitness in random_fitnesses]
    random_inter = [fitness[2] for fitness in random_fitnesses]
    plt.scatter(random_intra, random_inter, label='Random Pareto Front', color='green')

    # Highlight best solution
    plt.scatter(best_fitness[1], best_fitness[2], color='red', label='Best Solution (Max-Min Distance)')
    plt.xlabel('Intra Values')
    plt.ylabel('Inter Values')
    plt.title('Pareto Front: Intra vs. Inter Values')
    plt.legend()
    plt.show()

def plot_fitness_history(best_fitness_history, avg_fitness_history):
    generations = range(len(best_fitness_history))
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness_history, label='Best Fitness')
    plt.plot(generations, avg_fitness_history, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Modularity)')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_louvain(G, louvain_communities, save_file_path = None):
    node_to_community = {}  # Initialize the dictionary
    for community_id, community_nodes in enumerate(louvain_communities.communities):
        for node in community_nodes:
            node_to_community[node] = community_id

    # Assign colors based on community indices
    colors = [node_to_community[node] for node in G.nodes()]

    # Plot the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Layout for positioning nodes
    nx.draw(
        G, pos, node_color=colors, with_labels=True, cmap=plt.cm.rainbow, node_size=100
    )
    plt.title("Comunidades Detectadas pelo Algoritmo de Louvain")
    plt.show()

def visualize_comparison(graph, partition_ga, partition_louvain, nmi_score, save_file_path = None):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    pos = nx.spring_layout(graph, seed=42)

    # GA Communities Visualization
    communities_ga = defaultdict(list)
    for node, community in partition_ga.items():
        communities_ga[community].append(node)
    colors_ga = plt.cm.rainbow(np.linspace(0, 1, len(communities_ga)))
    color_map_ga = {}
    for color, nodes in zip(colors_ga, communities_ga.values()):
        for node in nodes:
            color_map_ga[node] = color
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=graph.nodes(),
                           node_color=[color_map_ga[node] for node in graph.nodes()], ax=axs[0])
    nx.draw_networkx_edges(graph, pos=pos, ax=axs[0])
    axs[0].set_title("MOCD - GA/Pareto")
    axs[0].axis('off')

    # Louvain Communities Visualization
    communities_louvain = {node: idx for idx, community in enumerate(partition_louvain.communities) for node in community}
    communities_louvain_dict = defaultdict(list)
    for node, community in communities_louvain.items():
        communities_louvain_dict[community].append(node)
    colors_louvain = plt.cm.rainbow(np.linspace(0, 1, len(communities_louvain_dict)))
    color_map_louvain = {}
    for color, nodes in zip(colors_louvain, communities_louvain_dict.values()):
        for node in nodes:
            color_map_louvain[node] = color
    nx.draw_networkx_nodes(graph, pos=pos, nodelist=graph.nodes(),
                           node_color=[color_map_louvain[node] for node in graph.nodes()], ax=axs[1])
    nx.draw_networkx_edges(graph, pos=pos, ax=axs[1])
    axs[1].set_title("Louvain Algorithm Communities")
    axs[1].axis('off')
    fig.suptitle(f'nmi_score: {nmi_score}', fontsize=16)
    
    if save_file_path == None:
        plt.show()
        return

    plt.savefig(save_file_path)