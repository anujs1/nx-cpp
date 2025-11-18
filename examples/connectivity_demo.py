import time

import networkx as nx
import nx_cpp


def main():
    print("=== Connected Components Demo ===")
    G = nx.Graph()
    G.add_edges_from(
        [
            (0, 1),
            (1, 2),
            (3, 4),
            (4, 5),
            (6, 7),
        ]
    )

    print("Small graph components (NetworkX vs nx-cpp union-find)")
    comps_py = [sorted(c) for c in nx.connected_components(G)]
    comps_cpp = [sorted(c) for c in nx.connected_components(G, backend="cpp")]
    print("NetworkX:", comps_py)
    print("nx-cpp  :", comps_cpp)

    print("\nSwitching to BFS-based components from nx-cpp")
    comps_bfs = [sorted(c) for c in nx_cpp.connected_components(G, method="bfs")]
    print("nx-cpp (BFS):", comps_bfs)

    print("\nTiming on a larger random graph")
    large = nx.gnp_random_graph(50_000, 0.0002, seed=2)
    t0 = time.time()
    _ = list(nx.connected_components(large))
    t1 = time.time()
    _ = list(nx.connected_components(large, backend="cpp"))
    t2 = time.time()
    _ = list(nx_cpp.connected_components(large, method="bfs"))
    t3 = time.time()

    print(f"NetworkX (py): {t1 - t0:.3f}s")
    print(f"nx-cpp (union-find): {t2 - t1:.3f}s")
    print(f"nx-cpp (BFS): {t3 - t2:.3f}s")


if __name__ == "__main__":
    main()
