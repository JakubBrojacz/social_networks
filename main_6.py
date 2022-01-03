import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from main_4 import power_dist_analysis


def ba_algorithm(m_0, m, n, draw_hist):
    if not draw_hist:
        fig, ax = plt.subplots(1, 1)
        xs = list(np.random.rand(m_0))
        ys = list(np.random.rand(m_0))
        num_repetitions = 1
    else:
        num_repetitions = 20
        degrees_summed = np.zeros(m_0+n, dtype=np.int32)

    for _ in tqdm(range(num_repetitions)):
        G = nx.complete_graph(m_0)
        degrees = [m_0-1 for _ in range(m_0)]
        for t in range(n):
            if not draw_hist:   
                ax.clear()
                ax.scatter(x=xs, y=ys, c='b')
                for e1, e2 in G.edges:
                    ax.plot([xs[e1], xs[e2]], [ys[e1], ys[e2]], color='k')
                plt.pause(0.001)
                xs.append((np.random.rand(1)-0.5)*np.log(len(xs)))
                ys.append((np.random.rand(1)-0.5)*np.log(len(ys)))

            new_node_id = m_0+t
            G.add_nodes_from([new_node_id])
            neighbors = np.random.choice(
                a=list(range(new_node_id)),
                size=m,
                replace=False,
                p=(np.asarray(degrees) / np.sum(degrees)))
            G.add_edges_from([(new_node_id, neigh) for neigh in neighbors])
            for neigh in neighbors:
                degrees[neigh] += 1
            degrees.append(m)
        if draw_hist:
            degrees_summed += np.asarray(degrees)

    if not draw_hist:
        plt.show()
        plt.close()
    else:
        n, _, _ = plt.hist(degrees_summed / num_repetitions, bins=30)
        plt.title("Degree distribution")
        plt.show()
        plt.close()
        plt.plot(range(degrees_summed.shape[0]), degrees_summed / num_repetitions)
        plt.plot(np.asarray(list(range(degrees_summed.shape[0]-1))) + m_0 + 1,
            m*np.sqrt(t/np.asarray(range(1, degrees_summed.shape[0]))))
        plt.title("Degree as a function of time of joining")
        plt.show()
        plt.close()
        power_dist_analysis(degrees_summed / num_repetitions)

    return G


def ba_algorithm_model_a(m_0, m, n, draw_hist):
    if not draw_hist:
        fig, ax = plt.subplots(1, 1)
        xs = list(np.random.rand(m_0))
        ys = list(np.random.rand(m_0))
        num_repetitions = 1
    else:
        num_repetitions = 20
        degrees_summed = np.zeros(m_0+n, dtype=np.int32)

    for _ in tqdm(range(num_repetitions)):
        G = nx.complete_graph(m_0)
        degrees = [m_0-1 for _ in range(m_0)]
        for t in range(n):
            if not draw_hist:   
                ax.clear()
                ax.scatter(x=xs, y=ys, c='b')
                for e1, e2 in G.edges:
                    ax.plot([xs[e1], xs[e2]], [ys[e1], ys[e2]], color='k')
                plt.pause(0.001)
                xs.append(np.random.rand(1))
                ys.append(np.random.rand(1))

            new_node_id = m_0+t
            G.add_nodes_from([new_node_id])
            neighbors = np.random.choice(
                a=list(range(new_node_id)),
                size=m,
                replace=False)
            G.add_edges_from([(new_node_id, neigh) for neigh in neighbors])
            for neigh in neighbors:
                degrees[neigh] += 1
            degrees.append(m)
        degrees_summed += np.asarray(degrees)

    if not draw_hist:
        plt.show()
        plt.close()
    else:
        n, _, _ = plt.hist(degrees_summed / num_repetitions, bins=30)
        xs = np.asarray(list(range(m, (np.max(degrees_summed) // num_repetitions) + 1)))
        ys = np.exp(1-xs/m)/m
        ys *= np.max(n) / np.max(ys)
        plt.plot(xs, ys)
        plt.title("Degree distribution")
        plt.show()
        plt.close()
        plt.plot(range(degrees_summed.shape[0]), degrees_summed / num_repetitions)
        plt.plot(np.asarray(list(range(degrees_summed.shape[0]-1))) + m_0 + 1,
            m+m*np.log(t/np.asarray(range(1, degrees_summed.shape[0]))))
        plt.title("Degree as a function of time of joining")
        plt.show()
        plt.close()
        power_dist_analysis(degrees_summed / num_repetitions)

    return G


if __name__ == '__main__':


    G = ba_algorithm(3, 2, 40, draw_hist=False)
    G = ba_algorithm(30, 20, 1000, draw_hist=True)
    G = ba_algorithm_model_a(30, 20, 1000, draw_hist=True)

