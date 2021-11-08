import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as special


def P3_1(mean=0, sigma=1):
    x = np.arange(mean-4*sigma, mean+4*sigma, 0.01)
    transformed_x = (x-mean)/sigma
    fx = np.exp(-transformed_x*transformed_x/2)/(sigma*np.sqrt(2*np.pi))
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, fx)
    sigma_positions = np.arange(-3, 4, 1)*sigma+mean
    for position in sigma_positions:
        ax.axvline(x=position, linewidth=1, color='r', linestyle='dashed')
    ax2 = ax.twiny()
    ax2.set_xticks(sigma_positions)
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(['-3σ', '-2σ', '-σ', 'mean', 'σ', '2σ', '3σ'])
    ax2.tick_params(axis='x', colors='red')
    plt.show()
    plt.close()

def P3_2a(p=0.5):
    x = np.arange(0, 100, 1)
    fx = np.power(1-p, x)*p
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, fx)
    pareto_share = 0.8
    cumulative_dist = np.cumsum(fx)
    for a, b in zip(x, cumulative_dist):
        if b>pareto_share:
            pareto_point = a
            break
    ax.axvline(x=pareto_point, linewidth=1, color='r', linestyle='dashed')
    ax2 = ax.twiny()
    ax2.set_xticks([pareto_point])
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(['80% of the distribution to the left'])
    ax2.tick_params(axis='x', colors='red')
    plt.title(f"Geometric distribution with p={p}")
    plt.tight_layout()
    plt.show()
    plt.close()


def P3_2b():
    p = np.arange(0.1, 1.0, 0.01)
    # 1 - (1 - p) ^ k >= 0.8
    # (1 - p) ^ k <= 0.2
    # k <= log{1-p} (0.2)
    # k <= log(0.2) / log(1-p)
    pareto_share = 0.8
    fp = np.ceil(np.log(1-pareto_share) / np.log(1-p))
    fig, ax = plt.subplots(1, 1)
    # ax.plot(p, fp)
    ax.scatter(p, fp, s=1)
    ax.set_xlabel("p")
    ax.set_ylabel("x of pareto point")
    ax.set_xticks(np.arange(0.1, 1.01, 0.1))
    plt.title(f"Geometric distribution")
    plt.tight_layout()
    plt.show()
    plt.close()


def P3_3(xmin=1, alpha=2):
    x = np.arange(xmin, xmin*8, xmin/10)
    fx = ((alpha-1)/xmin) * np.power(x/xmin, -alpha)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, fx)
    # sigma_positions = np.arange(-3, 4, 1)*sigma+mean
    # for position in sigma_positions:
    #     ax.axvline(x=position, linewidth=1, color='r', linestyle='dashed')
    # ax2 = ax.twiny()
    # ax2.set_xticks(sigma_positions)
    # ax2.set_xbound(ax.get_xbound())
    # ax2.set_xticklabels(['-3σ', '-2σ', '-σ', 'mean', 'σ', '2σ', '3σ'])
    # ax2.tick_params(axis='x', colors='red')
    plt.show()
    plt.close()


def P3_4(s=2):
    k = np.arange(1, 100, 1)
    fk = 1/(np.power(k, s)*special.zeta(s))
    fig, ax = plt.subplots(1, 1)
    ax.plot(k, fk)
    pareto_share = 0.8
    cumulative_dist = np.cumsum(fk)
    for a, b in zip(k, cumulative_dist):
        if b>pareto_share:
            pareto_point = a
            break
    ax.axvline(x=pareto_point, linewidth=1, color='r', linestyle='dashed')
    ax2 = ax.twiny()
    ax2.set_xticks([pareto_point])
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(['80% of the distribution to the left'])
    ax2.tick_params(axis='x', colors='red')
    ax.set_xlabel("k")
    ax.set_ylabel("PMF")
    plt.title(f"Zeta distribution with s={s}")
    plt.tight_layout()
    plt.show()
    plt.close()

def P5_5a():
    ns = [4, 8, 12]
    ms = [1, 2, 3, 4]
    fig, axs = plt.subplots(len(ns),len(ms), figsize=(15, 15))
    for i in range(len(ns)):
        n = ns[i]
        for j in range(len(ms)):
            m = min(n-1, ms[j])
            G = nx.generators.random_graphs.barabasi_albert_graph(n, m)
            nx.draw(G, ax=axs[i,j])
    plt.tight_layout()
    plt.show()
    plt.close()

def P5_5b():
    ns = [4, 8, 12]
    ps = [0.2, 0.4, 0.6, 0.8]
    fig, axs = plt.subplots(len(ns),len(ps), figsize=(15, 15))
    for i in range(len(ns)):
        n = ns[i]
        for j in range(len(ps)):
            p = min(n-1, ps[j])
            G = nx.generators.random_graphs.erdos_renyi_graph(n, p)
            nx.draw(G, ax=axs[i,j])
    plt.tight_layout()
    plt.show()
    plt.close()


def P5_5c(n=1000, m=50):
    G = nx.generators.random_graphs.barabasi_albert_graph(n, m)
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees)
    plt.title(f"Degree distribution in BA({n}, {m})")
    mean_d = np.mean(degrees)
    variance_d = np.mean(np.power(degrees - mean_d, 2))
    plt.suptitle(f"E(d)={mean_d} Var(d)={variance_d}")
    plt.axvline(x=mean_d, linewidth=1, color='r', linestyle='dashed')
    plt.show()
    plt.close()


def P5_5d(n=1000, p=0.05):
    G = nx.generators.random_graphs.erdos_renyi_graph(n, p)
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees)
    plt.title(f"Degree distribution in ER({n}, {p})")
    mean_d = np.mean(degrees)
    variance_d = np.mean(np.power(degrees - mean_d, 2))
    plt.suptitle(f"E(d)={mean_d} Var(d)={variance_d}")
    plt.axvline(x=mean_d, linewidth=1, color='r', linestyle='dashed')
    plt.show()
    plt.close()


def visualize(n, m):
    G = nx.generators.random_graphs.barabasi_albert_graph(n, m)
    num_nodes = G.number_of_nodes()
    edges = G.edges
    is_edge = np.zeros((num_nodes, num_nodes))
    for e1, e2 in edges:
        is_edge[e1, e2] = 1
        is_edge[e2, e1] = 1
    xs = np.random.uniform(low=0, high=1, size=num_nodes)
    ys = np.random.uniform(low=0, high=1, size=num_nodes)
    fig, ax = plt.subplots(1, 1)
    dt = 0.01
    electrostatic_strength = 10000
    for _ in range(1000):
        ax.clear()
        ax.scatter(x=xs, y=ys, c='b')
        for edge in edges:
            linexs = [xs[edge[0]], xs[edge[1]]]
            lineys = [ys[edge[0]], ys[edge[1]]]
            ax.plot(linexs, lineys, color='k')
        plt.pause(0.001)
        diffsx = np.asarray([xs-x for x in xs])
        diffsy = np.asarray([ys-y for y in ys])
        diffsx += np.eye(num_nodes)
        diffsy += np.eye(num_nodes)
        distances = np.sqrt(diffsx*diffsx+diffsy*diffsy)
        diffsx /= (distances)
        diffsy /= (distances)
        diffsx = diffsx - np.diag(np.diag(diffsx))
        diffsy = diffsy - np.diag(np.diag(diffsy))
        shift = dt * (electrostatic_strength / (distances*distances) - is_edge * (distances))
        xs += np.sum(diffsx * shift, axis=0)
        ys += np.sum(diffsy * shift, axis=0)

    plt.show()


if __name__ == '__main__':

    # P3_1(0, 1)
    # P3_2a(0.2)
    # P3_2b()
    # P3_3(100, 2)
    # P3_4(2)
    # P5_5a()
    # P5_5b()
    # P5_5c()
    # P5_5d()
    visualize(15, 14)