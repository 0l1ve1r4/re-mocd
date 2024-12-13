from genetic_operators import *

def genetic_algorithm(graph, population_size, generations):
    """Run a genetic algorithm to find the best partition."""
    population = initialize_population(graph, population_size)
    best_partition = None
    best_fitness = float('-inf')

    for i in range(generations):
        new_population = []
        fitness_scores = [(fitness_function(graph, partition)[0], partition) for partition in population]
        fitness_scores.sort(key=lambda x: x[0], reverse=True)

        if fitness_scores[0][0] > best_fitness:
            best_fitness = fitness_scores[0][0]
            best_partition = fitness_scores[0][1]

        for _ in range(population_size // 2):
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            child = two_point_crossover(parent1, parent2)
            if random.random() < 0.1:  # Mutation chance
                child = mutate(child, graph)
            new_population.append(child)

            
        debug(i, best_fitness)

        population = new_population

    
    return best_partition

def debug(generation: int, best_fitness: int) -> None:
    print( f"[GENERATION]: {generation}"
            "[BEST FITNESS]: {best_fitness}\n")