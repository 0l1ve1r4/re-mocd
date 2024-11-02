"""
ARTICLE_GRAPH = [
    (0,3), (0,7),
    (1,2), (1,3), (1,0), (1,7), (1,5),
    (2,3), (2,0),
    (4,5), (4,6),
    (5,6),
    (6,8),
    (7,8),
]
"""

import random

TEST_GRAPH = [
    # Comunidade 1
    (0, 1), (0, 2), (1, 2), (0, 3), (2, 3),
    
    # Comunidade 2
    (4, 5), (4, 6), (5, 6), (5, 7), (6, 7),
    
    # Comunidade 3
    (8, 9), (8, 10), (9, 10), (8, 11), (9, 11),
    
    # Comunidade 4
    (12, 13), (12, 14), (13, 14), (12, 15), (13, 15),
    
    # Comunidade 5
    (16, 17), (16, 18), (17, 18), (16, 19), (18, 19),
    
    # Conexões entre as comunidades
    (3, 4), (7, 8), (11, 12), (15, 16), (19, 0)
]

class GeneticAlgorithm():
    def __init__(self, graph: list, population_size: int):
        self.graph = graph
        self.nodes = list(set(node for edge in graph for node in edge))
        self.pop_size = population_size
        self.population = []

    def init(self):
        self.init_population()

    def get_adjacent_nodes(self, node):
        return [adj for edge in self.graph for adj in edge if node in edge and adj != node]

    def init_population(self):
        for _ in range(self.pop_size):
            individual = {}
            for node in self.nodes:
                adjacents = self.get_adjacent_nodes(node)
                individual[node] = random.choice(adjacents) if adjacents else node
            self.population.append(individual)
        print(self.population)

# ipsize and epsize are the size of IP and EP.
# p c and pm are the ratio of crossover and mutation.
# gen is the running generation.
#
# MOCD settles the following parameters: the ipsize and epsize
# both are 200, the gen is 500, pc and pm are 0.6 and 0.4, respectively.

POPULATION_SIZE = 1
PROBABILITY_CROSSOVER = 0.6
PROBABILITY_MUTATION = 0.4


if __name__ == "__main__":

    
    ga = GeneticAlgorithm(TEST_GRAPH, 1)
    ga.init()
