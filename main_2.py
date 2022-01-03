import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.swap as swap
import csv


def load_westeros():
    EDGES_PATH = './westeros/data/asoiaf-book1-edges.csv'
    edges_file = open(EDGES_PATH, "r")
    edges_reader = csv.reader(edges_file)
    edges_rows = [row for row in edges_reader][1:]
    nodes = list(set([row[0] for row in edges_rows]))
    edges = [(row[0], row[1], int(row[3])) for row in edges_rows]

    G = nx.MultiGraph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    return G


def load_unweighted_from_edges_file(path, no_header_lines=0):
    edges_file = open(path, "r")
    rows = [line[:-1].split(" ") for line in edges_file.readlines()[no_header_lines:]]
    nodes = list(set([int(row[0]) for row in rows]).union(set([int(row[1]) for row in rows])))
    edges = [(int(row[0]), int(row[1])) for row in rows]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def load_brain_cat():
    EDGES_PATH = './brain_cat/bn-cat-mixed-species_brain_1.edges'
    return load_unweighted_from_edges_file(EDGES_PATH)

def load_openflights():
    EDGES_PATH = './openflights/inf-openflights.edges'
    return load_unweighted_from_edges_file(EDGES_PATH, 2)

def load_socfb():
    EDGES_PATH = './socfb/socfb-American75.mtx'
    return load_unweighted_from_edges_file(EDGES_PATH, 2)

def load_diseasome():
    EDGES_PATH = './diseasome/bio-diseasome.mtx'
    return load_unweighted_from_edges_file(EDGES_PATH, 2)


def mean_nn_degree(G):
    degrees = {}
    for v in G.nodes():
        degree = 0
        for e1, e2 in G.edges([v]):
            degree += 1
        degrees[v] = degree
    degree_to_nn_degrees = [[] for _ in range(max(degrees.values())+1)]
    for v in G.nodes():
        nn_degrees = []
        for e1, e2 in G.edges([v]):
            if e1 != v:
                print("Ehh")
            nn_degrees.append(degrees[e2])
        mean_nn_degree = sum(nn_degrees) / len(nn_degrees)
        degree_to_nn_degrees[degrees[v]].append(mean_nn_degree)
    degree_to_mean_nn_degrees = [sum(x) / len(x) for x in degree_to_nn_degrees[1:] if len(x) > 0]
    plt.scatter(list(range(len(degree_to_mean_nn_degrees))), degree_to_mean_nn_degrees)
    plt.ylabel("Mean degree of nearest neighbour")
    plt.xlabel("Degree")
    plt.show()

if __name__ == '__main__':
    # G = load_westeros()
    # nx.write_gexf(G, "westeros.gexf")
    # G = load_brain_cat()
    # nx.write_gexf(G, "brain_cat.gexf")
    # G = load_openflights()
    # nx.write_gexf(G, "openflights.gexf")
    # # G = load_socfb()
    # # nx.write_gexf(G, "socfb.gexf")
    # G = load_diseasome()
    # nx.write_gexf(G, "diseasome.gexf")

    G = load_diseasome()
    mean_nn_degree(G)
    print(f"Pearson correlation coeff: {nx.algorithms.assortativity.degree_pearson_correlation_coefficient(G)}")

    while(True):
        swap.double_edge_swap(G, nswap=len(G.edges())/10, max_tries=10000000)
        mean_nn_degree(G)
        print(f"Pearson correlation coeff: {nx.algorithms.assortativity.degree_pearson_correlation_coefficient(G)}")
