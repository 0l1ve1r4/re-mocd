import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

data = "src/graphs/c_shi_2012/mocd_output.csv"
df = pd.read_csv(data)

group_size = 10
num_groups = len(df) // group_size
groups = [df.iloc[i * group_size:(i + 1) * group_size] for i in range(num_groups)]

group_names = [
    "Adjnoun", "Celegansmetabolic", "Celegansneural", "Football", 
    "Hepth", "Karate", "Lesmis", "Netscience", "Polbooks", "Power"
]

mean_times = []
ci_times = []
modularity_means = []
num_edges_means = []
num_nodes_means = []

for group in groups:
    elapsed_times = group['elapsed_time']
    modularity_values = group['modularity']
    num_edges = group['num_edges']
    num_nodes = group['num_nodes']

    mean_time = elapsed_times.mean()
    mean_modularity = modularity_values.mean()
    mean_edges = num_edges.mean()
    mean_nodes = num_nodes.mean()

    confidence_level = 0.95
    n = len(elapsed_times)
    se = elapsed_times.std(ddof=1) / np.sqrt(n)
    t_value = t.ppf((1 + confidence_level) / 2, n - 1)
    margin_error = t_value * se

    mean_times.append(mean_time)
    ci_times.append((mean_time - margin_error, mean_time + margin_error))
    modularity_means.append(mean_modularity)
    num_edges_means.append(mean_edges)
    num_nodes_means.append(mean_nodes)

overall_avg_time = np.mean(mean_times)
benchmark_time = 223

fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

indices = np.arange(len(group_names))
bar_width = 0.3

ax1.bar(indices - bar_width, mean_times, bar_width, 
        yerr=[[mean_time - ci[0] for mean_time, ci in zip(mean_times, ci_times)],
              [ci[1] - mean_time for mean_time, ci in zip(mean_times, ci_times)]],
        capsize=5, color='skyblue', label='Elapsed Time (s)')
ax1.axhline(benchmark_time, color='red', linestyle='--', label='Benchmark: 223 s')
ax1.text(len(mean_times) - 0.5, benchmark_time + 5, f'Overall Avg: {overall_avg_time:.2f} s', color='blue')
ax1.set_ylabel('Elapsed Time (s)', color='blue')
ax1.set_xlabel('Groups')
ax1.set_xticks(indices)
ax1.set_xticklabels(group_names, rotation=45)
ax1.tick_params(axis='y', labelcolor='blue')

for i, (edges, nodes) in enumerate(zip(num_edges_means, num_nodes_means)):
    ax1.text(i - bar_width, mean_times[i] + 2, f'E:{int(edges)}\nN:{int(nodes)}', 
             ha='center', color='black', fontsize=9)

ax3.scatter(indices, modularity_means, color='green', marker='o', s=100)
ax3.set_ylabel('Modularity', color='blue')
ax3.set_xticks(indices)
ax3.set_xticklabels(group_names, rotation=45)
ax3.tick_params(axis='x', labelsize=10)
ax3.tick_params(axis='y', labelcolor='green')

fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95), bbox_transform=ax1.transAxes)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.show()
