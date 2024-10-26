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
    def __init__(self, graph, ip_size=2, ep_size=5):
        self.graph = graph
        self.nodes = list(set(node for edge in graph for node in edge))
        self.ip_size = ip_size  
        self.ep_size = ep_size  
        self.internal_population = []  # IP
        self.external_population = []  # EP

    def init(self):
        self.init_internal_population()

    def get_adjacent_nodes(self, node):
        return [adj for edge in self.graph for adj in edge if node in edge and adj != node]

    def init_internal_population(self):
        for _ in range(self.ip_size):
            individual = {}
            for node in self.nodes:
                adjacents = self.get_adjacent_nodes(node)
                individual[node] = random.choice(adjacents) if adjacents else node
            self.internal_population.append(individual)
        print("Initial IP:", self.internal_population)

    def evaluate_solution(self, solution):
        # Função de avaliação hipotética (adapte conforme necessidade)
        return sum(solution.values())

    def update_external_population(self, candidate):
        # Avalia a solução e verifica se ela domina ou é dominada pelas soluções na EP
        candidate_score = self.evaluate_solution(candidate)
        to_remove = []
        
        # Remoção de soluções dominadas na EP
        for solution in self.external_population:
            if self.evaluate_solution(solution) < candidate_score:
                return  # Solução já é dominada, não adiciona
            elif self.evaluate_solution(solution) > candidate_score:
                to_remove.append(solution)  # Solução atual da EP é dominada, remover

        # Remove soluções dominadas e adiciona a nova solução
        for solution in to_remove:
            self.external_population.remove(solution)

        # Adiciona a nova solução se não dominada, respeitando o limite de EP
        if len(self.external_population) < self.ep_size:
            self.external_population.append(candidate)
        elif candidate_score < max(self.evaluate_solution(s) for s in self.external_population):
            # Substitui a pior solução se a EP estiver cheia
            worst_solution = max(self.external_population, key=self.evaluate_solution)
            self.external_population.remove(worst_solution)
            self.external_population.append(candidate)

        print("Updated EP:", self.external_population)

# Inicialização e exemplo de uso
ga = GeneticAlgorithm(TEST_GRAPH)
ga.init()
# Exemplo de atualização da EP com um novo indivíduo
new_individual = ga.internal_population[0]  # Apenas um exemplo, seria melhor gerar novos
ga.update_external_population(new_individual)
