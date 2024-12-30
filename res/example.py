import rmocd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def visualize_partition(graph: nx.Graph, partition_dict: dict):
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(16, 12)) 
    pos = nx.spring_layout(graph, seed=43)
    communities_dict = defaultdict(list)
    for node, community in partition_dict.items():
        communities_dict[community].append(node)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(communities_dict)))
    color_map = {
        node: color 
        for color, nodes in zip(colors, communities_dict.values()) 
        for node in nodes
    }

    nx.draw_networkx_edges(
        graph, pos=pos, ax=ax, alpha=0.7, width=3.0
    )
    
    nx.draw_networkx_nodes(
        graph, 
        pos=pos, 
        nodelist=graph.nodes(),
        node_color=[color_map[node] for node in graph.nodes()], 
        node_size=600,  
        ax=ax
    )
    
    nx.draw_networkx_labels(graph, pos=pos, ax=ax, font_size=10, font_color='black')
    
    for idx, (community, color) in enumerate(zip(communities_dict.values(), colors)):
        ax.scatter([], [], c=[color], label=f'Community {idx}', s=300) 
    
    ax.legend(scatterpoints=1, loc='upper left', fontsize=10, title="Communities")
    ax.axis('off')
    plt.savefig("example.png")
    plt.show()

def example_graph(save_path):
    G = nx.Graph()
    G.add_nodes_from(range(9))
    edges = [
        (0, 1), (0, 3), (0, 7),
        (1, 2), (1, 3), (1, 5),
        (2, 3),
        (3, 6),
        (4, 5), (4, 6),
        (5, 6),
        (7, 8)
    ]
    G.add_edges_from(edges)
    nx.write_edgelist(G, save_path, delimiter=",")
    return G


save_path = "example.edgelist"
G = example_graph(save_path)
G = nx.read_edgelist(save_path, delimiter=',', nodetype=int)

mocd_partition = rmocd.run(save_path)
visualize_partition(G, mocd_partition)