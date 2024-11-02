import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random



ARTICLE_GRAPH = [
    (0,3), (0,7),
    (1,2), (1,3), (1,0), (1,7), (1,5),
    (2,3), (2,0),
    (4,5), (4,6),
    (5,6),
    (6,8),
    (7,8),
]


G = nx.Graph()
G.add_edges_from(ARTICLE_GRAPH)

node_to_index = {node: i for i, node in enumerate(G.nodes())}
index_to_node = {i: node for node, i in node_to_index.items()}

POP_SIZE = 4
NUM_GENERATIONS = 500
MUTATION_RATE = 0.4     # p_m
CROSSOVER_RATE = 0.6    # p_c

def initialize_population(graph, pop_size):
    num_nodes = len(graph.nodes)
    return [np.random.randint(0, 2, num_nodes) for _ in range(pop_size)]

def modularity(graph, community):
    # Calculate modularity of the given community structure
    m = graph.number_of_edges()
    Q = 0
    for u in graph.nodes():
        for v in graph.nodes():
            if community[node_to_index[u]] == community[node_to_index[v]]:
                A_uv = 1 if graph.has_edge(u, v) else 0
                k_u = graph.degree[u]
                k_v = graph.degree[v]
                Q += A_uv - (k_u * k_v) / (2 * m)
    return Q / (2 * m)

def edge_cut(graph, community):
    # Calculate the number of edges between communities
    return sum(1 for u, v in graph.edges if community[node_to_index[u]] != community[node_to_index[v]])

def evaluate_population(graph, population):
    # Evaluate the modularity and edge cut for each solution in the population
    return [(modularity(graph, community), edge_cut(graph, community)) for community in population]

def select_parents(population, fitness):
    # Select individuals based on their fitness using a tournament selection
    idx1, idx2 = random.sample(range(len(population)), 2)
    return population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]

def crossover(parent1, parent2):
    # Single-point crossover between two parents
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(1, len(parent1) - 1)
        return np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])

    # if no one is going to cross over, select a random to survive:

    if random.random() < 0.5:
        return parent1
    
    else:
        return parent2

def mutate(individual):
    # Mutate an individual's community assignments with a small probability
    for i in range(len(individual)):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]  # Flip between community 0 and 1

def non_dominated_sorting(fitness):
    # Sort population based on Pareto dominance
    pareto_front = []
    for i, fi in enumerate(fitness):
        dominated = False
        for fj in fitness:
            if fj[0] >= fi[0] and fj[1] <= fi[1] and fj != fi:
                dominated = True
                break
        if not dominated:
            pareto_front.append(i)
    return pareto_front

# Main Genetic Algorithm loop
population = initialize_population(G, POP_SIZE)

print(f"POPULATION: {population}")

exit(0)

for generation in range(NUM_GENERATIONS):
    # Evaluate population
    fitness = evaluate_population(G, population)
    pareto_front = non_dominated_sorting(fitness)
    
    # Select next generation based on Pareto front
    new_population = []
    while len(new_population) < POP_SIZE:
        parent1 = select_parents(population, fitness)
        parent2 = select_parents(population, fitness)
        offspring = crossover(parent1, parent2)
        mutate(offspring)
        new_population.append(offspring)
    
    population = new_population

def show_community_graph(graph, community_assignments, title=""):
    plt.figure(figsize=(6, 6))
    
    # Assign colors to nodes based on their community
    community_colors = ["lightblue" if c == 0 else "lightcoral" for c in community_assignments]
    
    pos = nx.spring_layout(graph)  # Position nodes using a force-directed layout
    nx.draw_networkx(graph, pos, node_color=community_colors, with_labels=True, node_size=500, font_size=12)
    plt.title(title)
    plt.show()

# Final Pareto optimal solutions
final_fitness = evaluate_population(G, population)
pareto_front_indices = non_dominated_sorting(final_fitness)
pareto_solutions = [population[i] for i in pareto_front_indices]

# Output Pareto optimal solutions with graph visualization
for i, solution in enumerate(pareto_solutions):
    modularity = final_fitness[pareto_front_indices[i]][0]
    edge_cut = final_fitness[pareto_front_indices[i]][1]
    print(f"Solution {i+1}: Community assignments {solution}, Modularity = {modularity}, Edge Cut = {edge_cut}")
    
    # Visualize the graph for the current solution
    show_community_graph(G, solution, title=f"Solution {i+1}: Modularity = {modularity}, Edge Cut = {edge_cut}")