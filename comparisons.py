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
from tqdm import tqdm
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
import community as louvain 

#=========================================================================
# ~ Utils                                                                 
#=========================================================================

DEFAULT_NODES = 10000
NUM_GRAPHS = 10
RUNS_FOR_GRAPH = 1

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

def generate_random_community_sizes(n, min_size=10, max_size=100):
    """Generates random community sizes that sum up to `n`."""
    sizes = []
    remaining = n

    while remaining > 0:
        max_valid_size = min(max_size, remaining)  # Ensure we don't exceed `n`
        
        if max_valid_size < min_size:
            # If we can't allocate at least `min_size`, merge into the last community
            sizes[-1] += remaining
            break
        
        size = random.randint(min_size, max_valid_size)
        sizes.append(size)
        remaining -= size

    return sizes

def generate_lfr_graph(n, mu=0.1, seed=10, max_retries=100):
    """Generates an LFR benchmark graph with specific community sizes."""
    
    community_sizes = generate_random_community_sizes(n)
    if sum(community_sizes) != n:
        raise ValueError("Sum of community_sizes must equal n")

    retries = 0
    G = None

    while G is None and retries < max_retries:
        try:
            partition = {}
            node_id = 0
            for i, size in enumerate(community_sizes):
                for _ in range(size):
                    partition[node_id] = i
                    node_id += 1
            
            G = nx.LFR_benchmark_graph(
                n=n, 
                tau1=3, 
                tau2=1.5, 
                mu=mu, 
                average_degree=5, 
                max_degree=50,
                min_community=min(community_sizes),
                max_community=max(community_sizes),
                seed=seed
            )
        except Exception as e:
            print(e)
            print(f"Graph generation failed, retry {retries+1}/{max_retries}")
            retries += 1

    return G


# %%


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
def community_score(partition, G, r=1):
    def score(S):
        M_S = np.mean([(sum(1 for neighbor in G.neighbors(i) if neighbor in S) / len(S))**r for i in S])
        v_S = sum(1 for i in S for j in S if G.has_edge(i, j))
        return M_S * v_S
    
    return sum(score(S) for S in partition)

def community_fitness(partition, G, alpha=1):
    def P(S):
        return sum(sum(1 for neighbor in G.neighbors(i) if neighbor in S) / (sum(1 for neighbor in G.neighbors(i))**alpha) for i in S)
    
    return sum(P(S) for S in partition)

# Genetic Algorithm setup
def create_individual(G):
    return [random.choice(list(G.neighbors(node)) + [node]) for node in G.nodes()]

def decode_partition(individual, G):
    clusters = {}
    for node, link in zip(G.nodes(), individual):
        clusters.setdefault(link, []).append(node)
    return list(clusters.values())

def evaluate(individual, G):
    partition = decode_partition(individual, G)
    return -community_score(partition, G), -community_fitness(partition, G)

def run_moganet(G):
    start_time = time.time()
    
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: create_individual(G))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, G=G)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    
    pop = toolbox.population(n=100)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=300, lambda_=600, cxpb=0.8, mutpb=0.2, ngen=30, verbose=False)
    
    best_individual = tools.selBest(pop, 1)[0]
    best_partition = decode_partition(best_individual, G)
    
    return best_partition, time.time() - start_time


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
def run_louvain(G):
    start_time = time.time()
    
    # Convert igraph to NetworkX if needed
    if not isinstance(G, nx.Graph):
        G = nx.Graph(G.get_edgelist())
    
    # Get dictionary: node -> community label
    partition_dict = louvain.best_partition(G)
    louvain_time = time.time() - start_time
    
    # Convert dictionary to a set of frozensets (each representing a community)
    communities_dict = {}
    for node, comm in partition_dict.items():
        communities_dict.setdefault(comm, set()).add(node)
    partition = {frozenset(community) for community in communities_dict.values()}
    
    return partition, louvain_time


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

# %% [markdown]
# ### Experiments Functions

# %%
import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def experiment_worker(task):
    """
    Worker function to run a single experiment task.
    
    Parameters:
      task: tuple containing:
         - alg_name: Name of the algorithm (string)
         - alg_func: Function to run the algorithm
         - graph_idx: Index of the graph (integer)
         - run_idx: Run number (integer)
         - graph: The graph object
         - true_comms: The ground truth communities for the graph
         - graph_data: Dictionary of metadata for the graph
         
    Returns:
      A dictionary with the experiment results.
    """
    alg_name, alg_func, graph_idx, run_idx, graph, true_comms, graph_data = task
    
    # Run the algorithm
    comms, elapsed_time = alg_func(graph)
    nmi_val = evaluate_nmi(true_comms, comms, graph_data["Nodes"])
    result = {
        "Algorithm": alg_name,
        "Graph": graph_idx + 1,  # Convert 0-based index to 1-based
        "Run": run_idx + 1,
        f"{alg_name} Communities": len(comms),
        f"{alg_name} NMI": nmi_val,
        f"{alg_name} Time": elapsed_time,
    }
    result.update(graph_data)
    return result

def _run_experiment(generate_graph_func, output_filename, num_graphs, runs_per_graph):
    # --- 1. Load previous results if they exist ---
    if os.path.exists(output_filename):
        df_existing = pd.read_csv(output_filename)
        results = df_existing.to_dict("records")
        # Each completed experiment is identified by (Algorithm, Graph, Run)
        completed = set(zip(df_existing["Algorithm"], df_existing["Graph"], df_existing["Run"]))
    else:
        results = []
        completed = set()

    # --- 2. Generate graphs and precompute ground-truth communities ---
    lfr_graphs = []
    ground_truths = []
    graph_data_list = []
    
    for idx in tqdm(range(num_graphs), desc="Generating Graphs"):
        graph, n, mu = generate_graph_func(idx)
        lfr_graphs.append(graph)
        true_comms = {frozenset(graph.nodes[node]["community"]) for node in graph}
        ground_truths.append(true_comms)
        
        # Record metadata about the graph
        graph_data = {
            "Graph": idx + 1,
            "Nodes": n,
            "Edges": graph.number_of_edges(),
            "MU": mu,
            "Real Communities": len(true_comms),
        }
        graph_data_list.append(graph_data)

    # --- 3. Define the algorithms to run (one at a time) ---
    algorithms = {
        #"Leiden": run_leiden,
        
        #"MOCD": run_mocd,
        "REMOCD": run_remocd,
        #"MogaNet": run_moganet,
        "Louvain": run_louvain,
    }

    # --- 4. Run experiments for each algorithm in parallel ---
    for alg_name, alg_func in algorithms.items():
        print(f"\nRunning experiments for {alg_name}...")

        tasks = []
        for graph_idx in range(num_graphs):
            for run_idx in range(runs_per_graph):
                # Skip already completed tasks (identified by (Algorithm, Graph, Run))
                if (alg_name, graph_idx + 1, run_idx + 1) in completed:
                    continue
                task = (
                    alg_name,
                    alg_func,
                    graph_idx,
                    run_idx,
                    lfr_graphs[graph_idx],
                    ground_truths[graph_idx],
                    graph_data_list[graph_idx],
                )
                tasks.append(task)

        if not tasks:
            print(f"All experiments for {alg_name} are already completed.")
            continue
        
        max_workers = 10
        if alg_name == "REMOCD":
            max_workers = 1 # the algorithm is already parallelized internally
        with ProcessPoolExecutor(max_workers = max_workers) as executor:
            futures = {executor.submit(experiment_worker, task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {alg_name}"):
                try:
                    res = future.result()
                    results.append(res)
                    pd.DataFrame(results).to_csv(output_filename, index=False)
                except Exception as exc:
                    print(f"Task {futures[future]} generated an exception: {exc}")

    pd.DataFrame(results).to_csv(output_filename, index=False)
# %%
def node_size_based(mu=0.5, num_nodes=10000):
    """Evaluate algorithms with increasing node sizes."""
    def graph_generator(idx):
        multiplier = idx + 1
        n = num_nodes * multiplier
        graph = generate_lfr_graph(n=n, mu=mu)
        return graph, n, mu
    
    _run_experiment(
        generate_graph_func=graph_generator,
        output_filename="results_graph_size.csv",
        num_graphs=NUM_GRAPHS,
        runs_per_graph=RUNS_FOR_GRAPH,
    )

def mu_based(num_nodes=10000, base_mu=0.1):
    """Evaluate algorithms with increasing mixing parameter."""
    def graph_generator(idx):
        mu = base_mu * (idx + 1)
        graph = generate_lfr_graph(n=num_nodes, mu=mu)
        return graph, num_nodes, mu
    
    _run_experiment(
        generate_graph_func=graph_generator,
        output_filename="results_mu_metric.csv",
        num_graphs=NUM_GRAPHS,
        runs_per_graph=RUNS_FOR_GRAPH,
    )

node_size_based(num_nodes=DEFAULT_NODES)
#mu_based(num_nodes=DEFAULT_NODES)

# %% [markdown]
# # Plots

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('results_graph_size.csv')



sns.set_theme(style="whitegrid")

algorithms = ['Leiden', 'MOCD', 'REMOCD', 'Louvain']
times = {alg: df[f'{alg} Time'].dropna() for alg in algorithms}

means = [times[alg].mean() for alg in algorithms]
cis = [1.96 * times[alg].sem() for alg in algorithms]

plt.figure(figsize=(10, 6))
plt.bar(algorithms, means, yerr=cis, capsize=5, color='skyblue')
plt.xlabel('Algorithm')
plt.ylabel('Time (seconds)')
plt.title('Execution Time Comparison (95% CI)')
plt.show()

nmi_means = [df[f'{alg} NMI'].mean() for alg in algorithms]
nmi_cis = [1.96 * df[f'{alg} NMI'].sem() for alg in algorithms]
time_means = [df[f'{alg} Time'].mean() for alg in algorithms]
time_cis = [1.96 * df[f'{alg} Time'].sem() for alg in algorithms]

plt.figure(figsize=(10, 6))
for i, alg in enumerate(algorithms):
    plt.errorbar(time_means[i], nmi_means[i], xerr=time_cis[i], yerr=nmi_cis[i],
                 fmt='o', label=alg, markersize=8)
plt.xlabel('Time (seconds)')
plt.ylabel('NMI')
plt.xscale('log')
plt.title('NMI vs Time Trade-off (95% CI)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
for alg in algorithms:
    grouped = df.groupby('Graph').agg(
        mean_nmi=(f'{alg} NMI', 'mean'),
        nodes=('Nodes', 'first'),
        ci=(f'{alg} NMI', lambda x: 1.96 * x.sem())
    )
    plt.errorbar(grouped['nodes'], grouped['mean_nmi'], yerr=grouped['ci'],
                 fmt='o', label=alg, alpha=0.7)
plt.xlabel('Number of Nodes')
plt.ylabel('NMI')
plt.title('NMI vs Graph Size (95% CI)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
for alg in algorithms:
    grouped = df.groupby('Graph').agg(
        detected=(f'{alg} Communities', 'mean'),
        real=('Real Communities', 'first')
    )
    plt.scatter(grouped['real'], grouped['detected'], label=alg, alpha=0.6)

max_val = max(df['Real Communities'].max(), 
              max([df[f'{alg} Communities'].max() for alg in algorithms]))
plt.plot([0, max_val], [0, max_val], 'k--', label='Ideal')
plt.xlabel('Real Communities')
plt.ylabel('Detected Communities')
plt.title('Detected vs Real Communities')
plt.legend()
plt.grid(True)
plt.show()



# %%
