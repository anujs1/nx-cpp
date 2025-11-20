import random
import time

import networkx as nx


def make_weighted_graph(n=3000, p=0.003, seed=7):
    rng = random.Random(seed)
    G = nx.gnp_random_graph(n, p, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = rng.random() + 0.01
    return G


def total_weight(H):
    return sum(data.get("weight", 1.0) for _u, _v, data in H.edges(data=True))


def main():
    print("=== Minimum Spanning Tree Demo ===")
    G = make_weighted_graph()
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    t0 = time.time()
    mst_py = nx.minimum_spanning_tree(G, weight="weight")
    t1 = time.time()

    mst_cpp_kruskal = nx.minimum_spanning_tree(G, weight="weight", backend="cpp")
    t2 = time.time()

    mst_cpp_prim = nx.minimum_spanning_tree(G, weight="weight", backend="cpp", algorithm="prim")
    t3 = time.time()

    print(f"NetworkX (py, Kruskal): {t1 - t0:.3f}s")
    print(f"nx-cpp (Kruskal): {t2 - t1:.3f}s")
    print(f"nx-cpp (Prim): {t3 - t2:.3f}s")

    w_py = total_weight(mst_py)
    w_cpp_kruskal = total_weight(mst_cpp_kruskal)
    w_cpp_prim = total_weight(mst_cpp_prim)
    print(f"Total weight NetworkX: {w_py:.4f}")
    print(f"Total weight nx-cpp (Kruskal): {w_cpp_kruskal:.4f}")
    print(f"Total weight nx-cpp (Prim): {w_cpp_prim:.4f}")
    print("All match:", abs(w_py - w_cpp_kruskal) < 1e-6 and abs(w_py - w_cpp_prim) < 1e-6)


if __name__ == "__main__":
    main()
