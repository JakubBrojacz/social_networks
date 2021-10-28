import networkx as nx
import matplotlib.pyplot as plt


def P1_8(n):
    G = nx.Graph()
    G.add_nodes_from(range(n*n))
    G.add_edges_from([(n*row+col, n*row+col+1) for col in range(n-1) for row in range(n)])
    G.add_edges_from([(n*row+col, n*(row+1)+col) for col in range(n) for row in range(n-1)])
    return G


def P1_9(n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([(i, i+1) for i in range(n-1)])
    return G


def P1_10(n):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([(i, j) for i in range(n-1) for j in range(i+1, n)])
    return G


def P1_11(n):
    return nx.grid_graph([n, n])


def P1_12(n):
    return nx.path_graph(n)


def P1_13(n):
    return nx.complete_graph(n)


if __name__ == '__main__':

    n = 4

    G = P1_8(n)
    nx.draw(G)
    plt.show()
    plt.close()

    G = P1_9(n)
    nx.draw(G)
    plt.show()
    plt.close()

    G = P1_10(n)
    nx.draw(G)
    plt.show()
    plt.close()

    G = P1_11(n)
    nx.draw(G)
    plt.show()
    plt.close()

    G = P1_12(n)
    nx.draw(G)
    plt.show()
    plt.close()

    G = P1_13(n)
    nx.draw(G)
    plt.show()
    plt.close()
