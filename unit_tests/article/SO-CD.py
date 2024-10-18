import networkx as nx
import matplotlib.pyplot as plt

def partition_score(partition, graph):
    score = 0
    for community in partition:
        for i in range(len(community)):
            for j in range(i + 1, len(community)):
                node1 = community[i]
                node2 = community[j]
                if node2 in graph[node1]:  # Count edges within the community
                    score += 1
    return score

def greedy_community_detection(graph):
    nodes = list(graph.keys())
    partition = [[node] for node in nodes]

    improvement = True
    while improvement:
        improvement = False
        for node in nodes:
            best_community = None
            best_score = partition_score(partition, graph)
            for community in partition:
                if node not in community:
                    new_partition = [comm.copy() for comm in partition]
                    new_partition.remove(next(c for c in partition if node in c))
                    new_partition.append(community + [node])
                    score = partition_score(new_partition, graph)
                    if score < best_score:
                        best_score = score
                        best_community = new_partition

            if best_community:
                partition = best_community
                improvement = True

    return partition, partition_score(partition, graph)

def draw_graph(graph, partition):
    G = nx.Graph()  # Create an empty graph

    # Add edges to the graph based on the adjacency list
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    # Create a color map based on community membership
    color_map = []
    for node in G.nodes():
        for i, community in enumerate(partition):
            if node in community:
                color_map.append(i)  # Assign color based on community index

    # Draw the graph with node colors
    pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    nx.draw(G, pos, node_color=color_map, with_labels=True, cmap=plt.cm.rainbow)
    plt.title("Community Detection in Graph")
    plt.show()

# Example graph represented as a dictionary
graph = {
    0: [1, 2],
    1: [0, 2, 3],
    2: [0, 1, 4],
    3: [1, 4],
    4: [2, 3, 7],
    5: [6],
    6: [5],
    7: [8],
    8: [],
}

# Run community detection
best_partition, score = greedy_community_detection(graph)

# Print communities
print("Detected Communities:")
for i, community in enumerate(best_partition):
    print(f"Community {i + 1}: {community}")
print("Score (number of edges within communities):", score)

# Draw the graph
draw_graph(graph, best_partition)
