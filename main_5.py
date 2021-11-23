import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def adjacency_matrix(n, p):
    upper_t = np.triu((np.random.rand(n*n) < p).reshape(n, n), k=1)
    return (upper_t + upper_t.T).astype(np.int32)

def exercises():
    for n, p in [(200, 0.1), (200, 0.5), (1000, 0.5)]:
        am = adjacency_matrix(n, p)
        G = nx.convert_matrix.from_numpy_matrix(am)
        degrees = [G.degree(n) for n in G.nodes()]
        poisson_dist = [n]
        for k in range(1, max(degrees)):
            poisson_dist.append(poisson_dist[-1]*n*p/k)
        poisson_dist = np.exp(-n*p)*np.asarray(poisson_dist)
        plt.hist(degrees, bins=len(list(set(degrees))))
        plt.plot(poisson_dist)
        if n < 1000:
            avg_clustering = nx.average_clustering(G)
            plt.title(f"n={n}, p={p}\nclustering: {avg_clustering:.4f}")
        else:
            plt.title(f"n={n}, p={p}")

        plt.show()
        plt.close()

def P5_3():
    n = 20
    p_within = 0.8
    p_between = 0.1
    ran = np.random.rand(4*n*4*n).reshape(4*n,4*n)
    p = np.ones((4*n,4*n))*p_between
    small_square = np.ones((n,n))*p_within
    p[:n,:n] = small_square
    p[n:2*n,n:2*n] = small_square
    p[2*n:3*n,2*n:3*n] = small_square
    p[3*n:,3*n:] = small_square
    am = np.triu(ran < p, k=1)
    am = (am + am.T).astype(np.int32)
    G = nx.convert_matrix.from_numpy_matrix(am)
    nx.draw(G)
    plt.title("P5_3")
    plt.show()
    plt.close()

    nx.write_gexf(G, "main5data/withinbetween.gexf")

def P5_4():
    n = 1000
    k = 4
    xs = []
    ys = []
    for i in range(100):
        p = i/100
        G = nx.watts_strogatz_graph(n, k, p)
        avg_clustering = nx.average_clustering(G)
        xs.append(p)
        ys.append(avg_clustering)
        # print(f"p={p:.1f}: clustering {avg_clustering:.4f}")
    plt.scatter(xs, ys)
    plt.ylabel("average clustering coefficient")
    plt.xlabel("p")
    plt.title("P5_4")
    plt.show()
    plt.close()

def P5_5(n, e):
    lower_triange_size = (n*n-n)//2
    chosen_indices = np.random.choice(np.arange(lower_triange_size), size=e, replace=False)
    mask = np.asarray([1 if i in chosen_indices else 0 for i in range(lower_triange_size)])
    am = np.zeros((n,n), dtype=np.int32)
    am[np.triu_indices(n, 1)] = mask
    return am

def P5_6(degrees):
    am = np.zeros((len(degrees), len(degrees)), dtype=np.int32)
    while len(degrees) > 1 and degrees[-1] > 0:
        degree = degrees[-1]
        degrees = degrees[:-1]
        prev_max_degree_index = len(degrees)
        while(degree > 0):
            max_degree_index = degrees.index(max(degrees[:prev_max_degree_index]))
            for i in range(max_degree_index, prev_max_degree_index):
                am[i, len(degrees)] += 1
                am[len(degrees), i] += 1
                degrees[i] -= 1
                degree -= 1
                if degree == 0:
                    break
            prev_max_degree_index = max_degree_index

    if max(degrees) != 0 or min(degrees) != 0:
        return None
    return am

if __name__ == '__main__':
    exercises()

    P5_3()

    P5_4()

    am = P5_5(9, 12)
    G = nx.convert_matrix.from_numpy_matrix(am)
    nx.draw(G)
    plt.title("P5_5")
    plt.show()

    am = P5_6([2,3,3,4,4])
    if am is None:
        print("Failed to converge")
    else:
        G = nx.convert_matrix.from_numpy_matrix(am)
        nx.draw(G)
        plt.title("P5_6")
        plt.show()
        plt.close()
