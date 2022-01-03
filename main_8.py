import networkx as nx
from networkx.algorithms.bipartite.basic import color
import numpy as np
import matplotlib.pyplot as plt

from main_2 import load_brain_cat, load_diseasome

def get_c_k(G):
    M = nx.convert_matrix.to_numpy_matrix(G).astype(np.int32)
    max_degree = np.max(np.sum(M, axis=0))
    c_k = [[] for _ in range(max_degree+1)]
    for i in range(M.shape[0]):
        is_neighbour = (M[i] > 0)
        degree = np.sum(is_neighbour)
        if degree > 1:
            debug = np.multiply(M, is_neighbour.T)
            neigbour_connections = np.sum(np.multiply(debug, is_neighbour))
            c_k[degree].append(neigbour_connections / (degree * (degree-1)))
    return {i: np.mean(c_k[i]) for i in range(len(c_k)) if len(c_k[i]) > 0}


def load_bus_stop_positions(lines):
    start, end = None, None
    stop_positions = {}
    i = 0
    while True:
        if lines[i].find("*ZP") != -1:
            start = i
            break
        i += 1
    while True:
        if lines[i].find("#ZP") != -1:
            end = i
            break
        i += 1
    lines = lines[start+1:end]
    i = 0
    while i < len(lines):
        stop_id = lines[i][3:7]
        ys = []
        xs = []
        i += 1
        while True:
            y_coord_start = lines[i].find("Y=")
            if y_coord_start != -1:
                try:
                    ys.append(float(lines[i][y_coord_start+3:y_coord_start+12]))
                    x_coord_start = lines[i].find("X=")
                    xs.append(float(lines[i][x_coord_start+3:x_coord_start+12]))
                except:
                    pass
            if lines[i].find("#PR") != -1:
                break
            i +=1
        if len(xs) > 0:
            stop_positions[stop_id] = (sum(xs)/len(xs), sum(ys)/len(ys))
        i += 1

    return stop_positions


def load_bus_stop_edges(lines):
    start, end = None, None
    edges = []
    i = 0
    while True:
        if lines[i].find("*LL") != -1:
            start = i
            break
        i += 1
    while True:
        if lines[i].find("#LL") != -1:
            end = i
            break
        i += 1
    lines = lines[start+1:end]
    i = 0
    while i < len(lines):
        line_type = lines[i][17:-1]
        i += 1
        while True:
            prev_stop_id = None
            if lines[i].find("*LW") != -1:
                i += 1
                while True:
                    if lines[i].find("#LW") != -1:
                        break
                    stop_id = lines[i][49:53]
                    if stop_id != '    ':
                        if prev_stop_id is not None:
                            edges.append((prev_stop_id, stop_id, line_type))
                        prev_stop_id = stop_id
                    i += 1
            i += 1
            if lines[i].find("#TR") != -1:
                break
        while True:
            if lines[i].find("#WK") != -1:
                break
            i += 1
        i += 1
    return edges


if __name__ == '__main__':
    # G = load_diseasome()
    # c_k = get_c_k(G)
    # plt.plot(list(c_k.keys()), list(c_k.values()))
    
    # plt.show()
    data = open("RA211212.TXT")
    lines = data.readlines()

    positions = load_bus_stop_positions(lines)
    edges = load_bus_stop_edges(lines)

    fig, axs = plt.subplots(2, 2)
    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    axs[0,0].set_title("Tramwaje")
    axs[0,1].set_title("Zwykłe")
    axs[1,0].set_title("Nocne")
    axs[1,1].set_title("Inne")

    for e_start, e_end, e_type in edges:
        if e_start not in positions.keys():
            x_start = 21.017532
            y_start = 52.237049
        else:
            x_start = positions[e_start][0]
            y_start = positions[e_start][1]
        if e_end not in positions.keys():
            x_end = 21.017532
            y_end = 52.237049
        else:
            x_end = positions[e_end][0]
            y_end = positions[e_end][1]
        if e_type == 'LINIA TRAMWAJOWA':
            axs[0, 0].plot(
                [x_start, x_end],
                [y_start, y_end], color='red')
        elif e_type == 'LINIA ZWYKŁA':
            axs[0, 1].plot(
                [x_start, x_end],
                [y_start, y_end], color='blue')
        elif e_type == 'LINIA NOCNA':
            axs[1, 0].plot(
                [x_start, x_end],
                [y_start, y_end], color='black')
        else:
            axs[1, 1].plot(
                [x_start, x_end],
                [y_start, y_end], color='green')
    fig.text(0.5, 0.04, 'longitude', ha='center')
    fig.text(0.04, 0.5, 'latitude', va='center', rotation='vertical')
    plt.show()

    fig, axs = plt.subplots(2, 2)
    G = nx.Graph()
    chosen_edges = [(e[0], e[1]) for e in edges if e[2] == 'LINIA TRAMWAJOWA']
    nodes = list(set([e[0] for e in chosen_edges]).union(set([e[1] for e in chosen_edges])))
    G.add_nodes_from(nodes)
    G.add_edges_from(chosen_edges)
    c_k = get_c_k(G)
    axs[0,0].plot(list(c_k.keys()), list(c_k.values()))
    print("Tramwaje")
    print(f"Minimal degree: {min([x[1] for x in G.degree])}")
    print(f"Maximal degree: {max([x[1] for x in G.degree])}")
    print("Network density:", nx.density(G))
    print("Triadic closure:", nx.transitivity(G))

    G = nx.Graph()
    chosen_edges = [(e[0], e[1]) for e in edges if e[2] == 'LINIA ZWYKŁA']
    nodes = list(set([e[0] for e in chosen_edges]).union(set([e[1] for e in chosen_edges])))
    G.add_nodes_from(nodes)
    G.add_edges_from(chosen_edges)
    c_k = get_c_k(G)
    axs[0,1].plot(list(c_k.keys()), list(c_k.values()))
    print("\nZwykłe")
    print(f"Minimal degree: {min([x[1] for x in G.degree])}")
    print(f"Maximal degree: {max([x[1] for x in G.degree])}")
    print("Network density:", nx.density(G))
    print("Triadic closure:", nx.transitivity(G))

    G = nx.Graph()
    chosen_edges = [(e[0], e[1]) for e in edges if e[2] == 'LINIA NOCNA']
    nodes = list(set([e[0] for e in chosen_edges]).union(set([e[1] for e in chosen_edges])))
    G.add_nodes_from(nodes)
    G.add_edges_from(chosen_edges)
    c_k = get_c_k(G)
    axs[1,0].plot(list(c_k.keys()), list(c_k.values()))
    print("\nNocne")
    print(f"Minimal degree: {min([x[1] for x in G.degree])}")
    print(f"Maximal degree: {max([x[1] for x in G.degree])}")
    print("Network density:", nx.density(G))
    print("Triadic closure:", nx.transitivity(G))

    G = nx.Graph()
    chosen_edges = [(e[0], e[1]) for e in edges if e[2] not in ['LINIA TRAMWAJOWA', 'LINIA ZWYKŁA', 'LINIA NOCNA']]
    nodes = list(set([e[0] for e in chosen_edges]).union(set([e[1] for e in chosen_edges])))
    G.add_nodes_from(nodes)
    G.add_edges_from(chosen_edges)
    c_k = get_c_k(G)
    axs[1,1].plot(list(c_k.keys()), list(c_k.values()))
    print("\nInne")
    print(f"Minimal degree: {min([x[1] for x in G.degree])}")
    print(f"Maximal degree: {max([x[1] for x in G.degree])}")
    print("Network density:", nx.density(G))
    print("Triadic closure:", nx.transitivity(G))

    # G = nx.Graph()
    # chosen_edges = [(e[0], e[1]) for e in edges]
    # nodes = list(set([e[0] for e in chosen_edges]).union(set([e[1] for e in chosen_edges])))
    # G.add_nodes_from(nodes)
    # G.add_edges_from(chosen_edges)
    # c_k = get_c_k(G)
    # axs[1,1].plot(list(c_k.keys()), list(c_k.values()))
    # print("\nAll lines")
    # print(f"Minimal degree: {min([x[1] for x in G.degree])}")
    # print(f"Maximal degree: {max([x[1] for x in G.degree])}")
    # print("Network density:", nx.density(G))
    # print("Triadic closure:", nx.transitivity(G))

    axs[0,0].set_ylim([0.0, 1.0])
    axs[0,1].set_ylim([0.0, 1.0])
    axs[1,0].set_ylim([0.0, 1.0])
    axs[1,1].set_ylim([0.0, 1.0])
    axs[0,0].set_title("Tramwaje")
    axs[0,1].set_title("Zwykłe")
    axs[1,0].set_title("Nocne")
    axs[1,1].set_title("Inne")

    fig.text(0.5, 0.04, "vertex degree", ha='center')
    fig.text(0.04, 0.5, "mean clustering coefficient", va='center', rotation='vertical')
    plt.show()
