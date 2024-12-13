import random

def initialize_population(graph, size):
    """Generate a population of partitions based on adjacency."""
    population = []
    nodes = list(graph.nodes)

    for _ in range(size):
        partition = {}
        for node in nodes:
            adjacent = list(graph.neighbors(node))
            if adjacent:
                chosen_community = random.choice(adjacent)
                partition[node] = partition.get(chosen_community, chosen_community)
            else:
                partition[node] = node
        population.append(partition)

    return population

def two_point_crossover(parent1, parent2):
    """Perform two-point crossover on two parents."""
    nodes = list(parent1.keys())
    cut1, cut2 = sorted(random.sample(range(len(nodes)), 2))
    child = parent1.copy()

    for i in range(cut1, cut2):
        child[nodes[i]] = parent2[nodes[i]]

    return child

def mutate(partition, graph):
    """Perform a random mutation on a partition."""
    node = random.choice(list(partition.keys()))
    adjacent = list(graph.neighbors(node))
    if adjacent:
        partition[node] = random.choice(adjacent)
    return partition

def fitness_function(graph, partition):
    """Calculate the fitness of a partition based on the Rust-inspired Q(c)."""
    total_edges = graph.number_of_edges()
    if total_edges == 0:
        return 0.0, 0.0, 0.0

    intra_sum = 0.0
    inter = 0.0
    total_edges_doubled = 2.0 * total_edges

    community_groups = {}
    for node, community in partition.items():
        community_groups.setdefault(community, set()).add(node)

    for community in community_groups.values():
        community_edges = 0
        community_degree = 0.0

        for node in community:
            node_degree = len(list(graph.neighbors(node)))
            community_degree += node_degree

            for neighbor in graph.neighbors(node):
                if neighbor in community:
                    community_edges += 1

        community_edges /= 2
        intra_sum += community_edges
        normalized_degree = community_degree / total_edges_doubled
        inter += normalized_degree ** 2

    intra = 1.0 - (intra_sum / total_edges)
    modularity = 1.0 - intra - inter

    return modularity, intra, inter