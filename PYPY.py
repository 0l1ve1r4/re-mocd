# %% [markdown]
# # Imports

# %%
import numpy as np
import random
import networkx as nx
import pandas as pd
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from sklearn.metrics import normalized_mutual_info_score
import re_mocd as remocd
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from deap import base, creator, tools, algorithms

#=========================================================================
# ~ Utils                                                                 
#=========================================================================

NUM_GRAPHS = 5
RUNS_FOR_GRAPH = 1
DEFAULT_NODES = 250

def communities_to_labels(communities, n_nodes):
    labels = np.zeros(n_nodes, dtype=int)
    for comm_id, comm in enumerate(communities):
        for node in comm:
            labels[node] = comm_id
    return labels

def evaluate_nmi(true_communities, detected_communities, n_nodes):
    true_labels = communities_to_labels(true_communities, n_nodes)
    detected_labels = communities_to_labels(detected_communities, n_nodes)
    return normalized_mutual_info_score(true_labels, detected_labels)

def generate_lfr_graph(n, seed=10, min_community=20, mu=0.1, max_retries=100):
    G = None
    retries = 0

    while G is None and retries < max_retries:
        try:
            G = nx.LFR_benchmark_graph(
                n=n, 
                tau1=3, 
                tau2=1.5, 
                mu=mu, 
                average_degree=5, 
                min_community=min_community, 
                seed=seed
            )
        except Exception as e:
            print(f"Graph generation failed, min_community = {min_community + 10}")
            min_community += 10
            retries += 1
    return G

# %% [markdown]
# # Algorithms Definitions and functions

# %% [markdown]
# ### MOCD Model (Shi, 2012)

# %%
class MocdProblem(Problem):
    def __init__(self, G):
        self.G = G
        self.n_nodes = len(G)
        self.adjacency = [list(G.neighbors(node)) + [node] for node in G.nodes()]
        n_var = self.n_nodes
        xl = np.zeros(n_var, dtype=int)
        xu = np.array([len(neighbors) - 1 for neighbors in self.adjacency], dtype=int)
        super().__init__(n_var=n_var, n_obj=2, xl=xl, xu=xu, vtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        intra = []
        inter = []
        m = self.G.number_of_edges()
        
        for chrom in X:
            H = nx.Graph()
            for node, gene in enumerate(chrom):
                neighbor = self.adjacency[int(node)][int(gene)]
                H.add_edge(node, neighbor)
            
            communities = list(nx.connected_components(H))
            comms = [set(c) for c in communities]
            intra_val = sum(sum(1 for u, v in self.G.edges(c) if v in c) for c in comms) / m
            inter_val = sum((sum(self.G.degree(u) for u in c) / (2 * m)) ** 2 for c in comms)
            
            intra.append(1 - intra_val)
            inter.append(inter_val)
        
        out["F"] = np.column_stack([intra, inter])

def max_min_selection(real_front, random_front):
    min_distances = []
    for real_sol in real_front:
        distances = [np.linalg.norm(real_sol - random_sol) for random_sol in random_front]
        min_distances.append(np.min(distances))
    return np.argmax(min_distances)

def decode_communities(chromosome, problem):
    H = nx.Graph()
    for node, gene in enumerate(chromosome):
        neighbor = problem.adjacency[int(node)][int(gene)]
        H.add_edge(node, neighbor)
    return list(nx.connected_components(H))

def run_mocd(G):
    start_time = time.time()
    problem = MocdProblem(G)
    algorithm = NSGA2(
        pop_size=100,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )
    res = minimize(problem, algorithm, ('n_gen', 100), verbose=False)
    
    # Generate random graph with same degree sequence
    degrees = [d for _, d in G.degree()]
    random_G = nx.configuration_model(degrees)
    nx.relabel_nodes(random_G, {v: k for k, v in enumerate(random_G.nodes())}, copy=False)
    random_problem = MocdProblem(random_G)
    random_res = minimize(random_problem, algorithm, ('n_gen', 100), verbose=False)
    
    # Model selection
    selected_idx = max_min_selection(res.F, random_res.F)
    best_solution = res.X[selected_idx]
    communities = decode_communities(best_solution, problem)
    
    return communities, time.time() - start_time

# %% [markdown]
# ### MOGA-Net (Pizzuti, 2009)

# %%
def community_score(partition, G):
    score = 0
    for community in partition:
        vS = sum(1 for i in community for j in community if G.has_edge(i, j))
        mu = [sum(1 for j in community if G.has_edge(i, j)) / len(community) for i in community]
        M_S = sum(m ** 2 for m in mu) / len(community) if len(community) > 0 else 0
        score += M_S * vS
    return score,

def community_fitness(partition, G, alpha=1):
    fitness = 0
    for community in partition:
        kin = {i: sum(1 for j in community if G.has_edge(i, j)) for i in community}
        kout = {i: sum(1 for j in G.nodes if j not in community and G.has_edge(i, j)) for i in community}
        P_S = sum(kin[i] / ((kin[i] + kout[i]) ** alpha) if (kin[i] + kout[i]) > 0 else 0 for i in community)
        fitness += P_S
    return fitness,

# Genetic Algorithm Setup
def individual_to_partition(individual, G):
    mapping = {i: individual[i] for i in range(len(individual))}
    clusters = {}
    for node, leader in mapping.items():
        clusters.setdefault(leader, set()).add(node)
    return list(clusters.values())

def evaluate(individual, G):
    partition = individual_to_partition(individual, G)
    return community_score(partition, G)[0], community_fitness(partition, G)[0]

def run_moganet(G):
    start_time = time.time()
    
    num_nodes = len(G.nodes)
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_node", random.randint, 0, num_nodes - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_node, num_nodes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_nodes-1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate, G=G)
    
    pop = toolbox.population(n=100)
    hof = tools.ParetoFront()
    
    algorithms.eaMuPlusLambda(pop, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.3, ngen=50, stats=None, halloffame=hof, verbose=False)
    
    best_partition = individual_to_partition(hof[0], G)
    elapsed_time = time.time() - start_time
    
    return best_partition, elapsed_time

# %% [markdown]
# ### RE-MOCD

# %%

def run_remocd(G):
    start_time = time.time()
    community_dict = remocd.from_nx(G, debug=False)
    remocd_time = time.time() - start_time
    
    # dict -> list of sets
    community_map = {}
    for node, cid in community_dict.items():
        if cid not in community_map:
            community_map[cid] = set()
        community_map[cid].add(node)
    return list(community_map.values()), remocd_time

# %% [markdown]
# ### Leiden/Louvain

# %%
def run_leiden(G):
    import leidenalg
    from igraph import Graph  
    start_time = time.time()
    
    # Convert NetworkX graph to igraph format if needed
    if not isinstance(G, Graph):
        G = Graph.from_networkx(G)  
    
    partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
    leiden_time = time.time() - start_time
    return partition, leiden_time

# %% [markdown]
# # Running the Experiments

# %%
results_same = []

for i in range(1, NUM_GRAPHS + 1):
    base_graph = generate_lfr_graph(n=DEFAULT_NODES, seed=10, min_community=20, mu=0.1 * i)
    true_communities_base = {frozenset(base_graph.nodes[node]["community"]) for node in base_graph}

    for same_graph in range(RUNS_FOR_GRAPH):
        print(f"Graph {i}/{NUM_GRAPHS} - Run: {same_graph}/{RUNS_FOR_GRAPH}")

        print("[DEBUG]: Running leiden")
        leiden_comms, leiden_time = run_leiden(base_graph)
        
        print("\033[F\033[K", end="") 
        print("[DEBUG]: Running mocd")
        mocd_comms, mocd_time = run_mocd(base_graph)
        print("\033[F\033[K", end="") 
        print("[DEBUG]: Running remocd")
        remocd_comms, remocd_time = run_remocd(base_graph)
        print("\033[F\033[K", end="") 
        print("[DEBUG]: Running moganet")
        moganet_comms, moganet_time = run_moganet(base_graph)

        print("\033[F\033[K", end="") 
        print("[DEBUG]: Evaluating NMI")
        nmi_mocd = evaluate_nmi(true_communities_base, mocd_comms, len(base_graph))
        nmi_remocd = evaluate_nmi(true_communities_base, remocd_comms, len(base_graph))
        nmi_moganet = evaluate_nmi(true_communities_base, moganet_comms, len(base_graph))
        nmi_leiden = evaluate_nmi(true_communities_base, leiden_comms, len(base_graph))

        results_same.append({
            "Graph": i,
            "Nodes": len(base_graph),
            "Edges": base_graph.number_of_edges(),
            "MU": 0.1 * i,
            "Real Communities": len(true_communities_base),

            "MOCD Communities": len(mocd_comms),
            "MOCD NMI": nmi_mocd,
            "MOCD Time": mocd_time,

            "REMOCD Communities": len(remocd_comms),
            "REMOCD NMI": nmi_remocd,
            "REMOCD Time": remocd_time,
        })

df_same = pd.DataFrame(results_same)
df_same.to_csv("mocd_vs_remocd_same_graph.csv", index=False)


# %% [markdown]
# # Plots

# %%
df = pd.read_csv('mocd_vs_remocd_same_graph.csv')

# ============================================================
# MU plot
# ============================================================

nmi_df = df.melt(
    id_vars=['MU'], 
    value_vars=['MOCD NMI', 'REMOCD NMI'], 
    var_name='Method', 
    value_name='NMI'
)
nmi_df['Method'] = nmi_df['Method'].str.replace(' NMI', '')

# Prepare Time data in long format
time_df = df.melt(
    id_vars=['MU'], 
    value_vars=['MOCD Time', 'REMOCD Time'], 
    var_name='Method', 
    value_name='Time'
)
time_df['Method'] = time_df['Method'].str.replace(' Time', '')

# Plot NMI comparison
plt.figure(figsize=(10, 6))
sns.lineplot(data=nmi_df, x='MU', y='NMI', hue='Method', errorbar=('ci', 95))
plt.title('NMI Comparison between MOCD and REMOCD')
plt.xlabel('MU')
plt.ylabel('NMI')
plt.grid(True)
plt.show()

# Plot Time comparison with log scale
plt.figure(figsize=(10, 6))
sns.lineplot(data=time_df, x='MU', y='Time', hue='Method', ci=95)
plt.title('Time Comparison between MOCD and REMOCD')
plt.xlabel('MU')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.show()


