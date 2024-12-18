import networkx as nx
from multiprocessing import Pool
import matplotlib.pyplot as plt
import random
import numpy as np
import math

from collections import defaultdict

def calculate_objectives(graph, partition) -> (float, float, float):
    total_edges = graph.number_of_edges()
    if total_edges == 0:
        return 0.0, 0.0, 0.0

    intra_sum = 0.0
    inter = 0.0
    total_edges_doubled = 2.0 * total_edges

    communities = defaultdict(set)
    for node, community in partition.items():
        communities[community].add(node)

    for community_nodes in communities.values():
        community_edges = 0
        community_degree = 0.0

        for node in community_nodes:
            node_degree = len(list(graph.neighbors(node)))
            community_degree += node_degree

            for neighbor in graph.neighbors(node):
                if neighbor in community_nodes:
                    community_edges += 1

        if not graph.is_directed():
            community_edges /= 2

        intra_sum += community_edges
        normalized_degree = community_degree / total_edges_doubled
        inter += normalized_degree ** 2

    intra = 1.0 - (intra_sum / total_edges)
    modularity = 1.0 - intra - inter
    modularity = max(-1.0, min(1.0, modularity))

    return modularity, intra, inter

# Generate initial population
def generate_initial_population(graph, population_size):
    population = []
    nodes = list(graph.nodes())
    for _ in range(population_size):
        partition = {node: random.randint(0, len(nodes)-1) for node in nodes}
        population.append(partition)
    return population

# Two-point crossover
def crossover(parent1, parent2):
    keys = list(parent1.keys())
    idx1, idx2 = sorted(random.sample(range(len(keys)), 2))
    child = parent1.copy()
    for i in range(idx1, idx2):
        child[keys[i]] = parent2[keys[i]]
    return child

# Mutation ensuring mutations only between adjacent nodes
def mutate(partition, graph):
    node = random.choice(list(partition.keys()))
    neighbors = list(graph.neighbors(node))
    if neighbors:
        partition[node] = partition[random.choice(neighbors)]
    return partition

# Selection based on Pareto dominance (simplified)
def selection(population, fitnesses):
    # Sort based on modularity
    sorted_population = [p for _, p in sorted(zip(fitnesses, population), key=lambda x: x[0][0], reverse=True)]
    return sorted_population[:len(population)//2]

# Determine if solution1 dominates solution2
def dominates(solution1, solution2):
    """Checks if solution1 dominates solution2."""
    return all(s1 <= s2 for s1, s2 in zip(solution1, solution2)) and any(s1 < s2 for s1, s2 in zip(solution1, solution2))

# Compute crowding distance for a Pareto front
def compute_crowding_distance(front):
    """Computes the crowding distance for a set of solutions in a Pareto front."""
    size = len(front)
    if size == 0:
        return []

    # Initialize distances to zero
    distances = [0] * size

    # Normalize each objective
    num_objectives = len(front[0])
    for i in range(num_objectives):
        # Sort by the i-th objective
        sorted_front = sorted(front, key=lambda x: x[i])
        min_value = sorted_front[0][i]
        max_value = sorted_front[-1][i]

        # Avoid division by zero
        if max_value - min_value == 0:
            continue

        # Assign infinite distance to boundary solutions
        distances[0] = distances[-1] = float("inf")

        # Compute distances for interior solutions
        for j in range(1, size - 1):
            distances[j] += (sorted_front[j + 1][i] - sorted_front[j - 1][i]) / (max_value - min_value)

    return distances

# Fitness-Based Selection
def fitness_based_selection(population, fitnesses, num_selected):
    """Selects individuals using Pareto dominance and crowding distance."""
    # Step 1: Group solutions into Pareto fronts
    pareto_fronts = []
    remaining_population = set(range(len(population)))
    while remaining_population:
        current_front = []
        for i in remaining_population:
            if not any(dominates(fitnesses[j], fitnesses[i]) for j in remaining_population if j != i):
                current_front.append(i)
        pareto_fronts.append(current_front)
        remaining_population -= set(current_front)

    # Step 2: Select solutions from Pareto fronts
    selected = []
    for front in pareto_fronts:
        if len(selected) + len(front) <= num_selected:
            selected.extend(front)
        else:
            # Compute crowding distance for the current front
            front_solutions = [fitnesses[i] for i in front]
            crowding_distances = compute_crowding_distance(front_solutions)
            # Sort by crowding distance in descending order
            sorted_front = sorted(zip(front, crowding_distances), key=lambda x: x[1], reverse=True)
            # Fill remaining slots
            selected.extend([idx for idx, _ in sorted_front[:num_selected - len(selected)]])
            break

    return [population[i] for i in selected]

# Calculate the distance between two solutions
def calculate_distance(fitness1, fitness2):
    intra_diff = fitness1[1] - fitness2[1]
    inter_diff = fitness1[2] - fitness2[2]
    distance = math.sqrt(intra_diff ** 2 + inter_diff ** 2)
    return distance

# Genetic algorithm function with Max-Min Distance selection
def genetic_algorithm(graph, generations=80, population_size=100):
    best_fitness_history = []
    avg_fitness_history = []
    
    # Run on real network
    real_population = generate_initial_population(graph, population_size)
    for generation in range(generations):
        with Pool() as pool:
            # Create a list of arguments for each call
            args = [(graph, partition) for partition in real_population]
            fitnesses = pool.starmap(calculate_objectives, args)

        modularity_values = [fitness[0] for fitness in fitnesses]
        best_fitness = max(modularity_values)
        avg_fitness = sum(modularity_values) / len(modularity_values)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)


        # Select individuals for mating
        # Use the new selection function
        real_population = selection(real_population, fitnesses)
        # real_population = fitness_based_selection(real_population, fitnesses, num_selected=len(real_population)//2)
        # Generate new population
        new_population = []
        while len(new_population) < population_size:
            parents = random.sample(real_population, 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child, graph)
            new_population.append(child)
        real_population = new_population

        print(f"[Generation]: {generation}")

    # Final evaluation for real network
    real_fitnesses = [calculate_objectives(graph, partition) for partition in real_population]
    real_pareto_front = [partition for fitness, partition in zip(real_fitnesses, real_population) if fitness[0] == max(real_fitnesses, key=lambda x: x[0])[0]]

    # Run on random network
    random_graph = nx.gnm_random_graph(graph.number_of_nodes(), graph.number_of_edges())
    random_population = generate_initial_population(random_graph, population_size)
    for generation in range(generations):
        # Evaluate fitness
        fitnesses = [calculate_objectives(random_graph, partition) for partition in random_population]
        # Select individuals for mating
        random_population = selection(random_population, fitnesses)
        # Generate new population
        new_population = []
        while len(new_population) < population_size:
            parents = random.sample(random_population, 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child, random_graph)
            new_population.append(child)
        random_population = new_population

    # Final evaluation for random network
    random_fitnesses = [calculate_objectives(random_graph, partition) for partition in random_population]
    random_pareto_front_fitnesses = [fitness for fitness in random_fitnesses if fitness[0] == max(random_fitnesses, key=lambda x: x[0])[0]]

    # Max-Min Distance Selection
    max_deviation = -1
    best_partition = None
    deviations = []
    for real_partition, real_fitness in zip(real_pareto_front, real_fitnesses):
        # Calculate deviation of this solution
        min_distance = min([calculate_distance(real_fitness, random_fitness) for random_fitness in random_pareto_front_fitnesses])
        deviations.append((real_partition, real_fitness, min_distance))
        if min_distance > max_deviation:
            max_deviation = min_distance
            best_partition = real_partition
            best_fitness = real_fitness

    return best_partition, deviations, real_fitnesses, random_fitnesses, best_fitness_history, avg_fitness_history
