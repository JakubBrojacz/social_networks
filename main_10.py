import csv
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.relabel import convert_node_labels_to_integers

class static_variables:
    NUM_COMMUNITIES = 2

def generate_example_community_graph(num_nodes, num_communities, k_within, k_between):
    adj_matrix = np.zeros((num_nodes, num_nodes))
    community_size = num_nodes // num_communities
    for i in range(num_nodes):
        community = i // community_size
        community_start = community * community_size
        community_end = (community + 1) * community_size
        chosen_neighbors = np.random.choice(list(range(community_start, community_end)), k_within)
        for neigh in chosen_neighbors:
            if neigh > i:
                adj_matrix[i, neigh] = 1
                adj_matrix[neigh, i] = 1
        chosen_neighbors = np.random.choice(list(range(0, community_start)) + list(range(community_end, num_nodes)), k_between)
        for neigh in chosen_neighbors:
            if neigh > i:
                adj_matrix[i, neigh] = 1
                adj_matrix[neigh, i] = 1
    return nx.from_numpy_array(adj_matrix)


def get_best_partition(G, node_list):
    if len(node_list) == 1:
        return None

    num_nodes = len(G.nodes)
    communities = np.random.randint(0, 2, size=num_nodes)
    kappas = np.zeros(num_nodes)
    aris = np.zeros(2)
    degrees = np.zeros(num_nodes)

    for node in node_list:
        node_community = communities[node]
        for edge in G.edges(node):
            degrees[node] += 1
            if communities[edge[1]] == node_community:
                kappas[node] += 1
                aris[node_community] += 1
            else:
                aris[node_community] += 2

    aris /= 2
    num_edges = np.sum(degrees) / 2
    best_communities = communities.copy()
    global_fitness = np.sum([kappas[i]*num_edges/degrees[i] - aris[communities[i]] for i in node_list])
    best_fitness = global_fitness
    num_tries = 0

    while num_tries < len(node_list):
        num_tries += 1
        worst_nodes = np.argsort([kappas[i]*num_edges/degrees[i] - aris[communities[i]] for i in node_list])
        probabilities = 1/np.power(np.arange(1, worst_nodes.shape[0]+1), 1.8)
        node = node_list[np.random.choice(worst_nodes, 1, p=probabilities/np.sum(probabilities))[0]]
        node = node_list[worst_nodes[0]]
        communities[node] = 1-communities[node]
        kappas[node] = degrees[node]-kappas[node]
        for edge in G.edges(node):
            neighbor = edge[1]
            if communities[neighbor] == communities[node]:
                kappas[neighbor] += 1
                aris[1-communities[node]] -= 1
            else:
                kappas[neighbor] -= 1
                aris[communities[node]] += 1

        global_fitness = np.sum([kappas[i]*num_edges/degrees[i] - aris[communities[i]] for i in node_list])

        if global_fitness > best_fitness:
            best_communities = communities.copy()
            best_fitness = global_fitness
            num_tries = 0

    communities = best_communities

    if best_fitness <= 0:
        return None

    community_0_node_list = [i for i in node_list if communities[i] == 0]
    community_1_node_list = [i for i in node_list if communities[i] == 1]

    if len(community_0_node_list) > 0 and len(community_1_node_list) > 0:
        for node in node_list:
            for edge in list(G.edges(node)):
                if communities[edge[1]] != communities[node]:
                    G.remove_edge(edge[0], edge[1])

        communities += static_variables.NUM_COMMUNITIES
        static_variables.NUM_COMMUNITIES += 2

        communities_0 = get_best_partition(G, community_0_node_list)
        if communities_0 is not None:
            for i in community_0_node_list:
                communities[i] = static_variables.NUM_COMMUNITIES + communities_0[i]
            static_variables.NUM_COMMUNITIES += 2
        communities_1 = get_best_partition(G, community_1_node_list)
        if communities_1 is not None:
            for i in community_1_node_list:
                communities[i] = 1 + static_variables.NUM_COMMUNITIES + communities_1[i]
            static_variables.NUM_COMMUNITIES += 3
    else:
        communities += static_variables.NUM_COMMUNITIES
        static_variables.NUM_COMMUNITIES += 2

    return communities

if __name__ == '__main__':
    # G_original = nx.barbell_graph(20, 3)
    G_original = generate_example_community_graph(400, 8, 40, 4)
    print(f"Number of nodes: {len(G_original.nodes)}")
    G = convert_node_labels_to_integers(G_original)
    np.random.seed(20)

    communities = get_best_partition(G, list(range(len(G.nodes))))
    if communities is None:
        communities = np.zeros(len(G.nodes))
    unique_communities = np.unique(communities)
    communities = [np.argwhere(unique_communities == c)[0][0] for c in communities]

    colors = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "grey", 5: "deeppink", 6: "black", 7: "yellow"}
    color_map = [colors[c%8] for c in communities]
    pos = nx.spring_layout(G_original, seed=42)
    np.random.seed(42)
    nx.draw(G_original, node_color=color_map, pos=pos)
    plt.show()
