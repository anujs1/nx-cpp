import time
import networkx as nx
import random


def main():
    # Create a large grid graph for shortest path testing
    # Grid graphs force longer paths between corners
    # Create a large 2D grid (like a city street network)
    print("Creating weighted grid graph...")
    grid_size = 500  # 500x500 = 250,000 nodes
    print(f"Building {grid_size}x{grid_size} grid...")
    t_graph_start = time.time()
    G = nx.grid_2d_graph(grid_size, grid_size)
    random.seed(42)
    for u, v in G.edges():
        G[u][v]["weight"] = random.uniform(1.0, 10.0)
    t_graph_end = time.time()
    print(f"Graph creation time: {t_graph_end - t_graph_start:.3f}s")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Pick diagonal corners for maximum path length
    source = (0, 0)
    target = (grid_size - 1, grid_size - 1)
    print(f"\nFinding shortest path from {source} to {target}")

    print("=" * 50)
    print("Dijkstra's Algorithm")
    print("=" * 50)
    # Dijkstra with NetworkX (Python)
    t0 = time.time()
    try:
        path_py = nx.shortest_path(G, source, target, weight="weight", method="dijkstra")
        path_len_py = len(path_py)
    except nx.NetworkXNoPath:
        path_py = None
        path_len_py = 0
    t1 = time.time()

    # Dijkstra with C++ backend (includes conversion time)
    try:
        path_cpp = nx.shortest_path(G, source, target, weight="weight", method="dijkstra", backend="cpp")
        path_len_cpp = len(path_cpp)
    except nx.NetworkXNoPath:
        path_cpp = None
        path_len_cpp = 0
    t2 = time.time()

    print(f"NetworkX (py): {t1 - t0:.3f}s, path length: {path_len_py}")
    print(f"nx-cpp backend (includes conversion): {t2 - t1:.3f}s, path length: {path_len_cpp}")
    print(f"Speedup: {(t1 - t0)/(t2 - t1):.2f}x")
    same_length = path_len_py == path_len_cpp
    print(f"\nVerification: Both paths have length {path_len_py}: {same_length}")

    if path_py and path_cpp:
        print(f"Python path starts: {path_py[:5]}")
        print(f"C++ path starts: {path_cpp[:5]}")

    print("\n" + "=" * 50)
    print("Bellman-Ford Algorithm")
    print("=" * 50)

    # Bellman-Ford with NetworkX (Python)
    t0 = time.time()
    try:
        path_py_bf = nx.shortest_path(G, source, target, weight="weight", method="bellman-ford")
        path_len_py_bf = len(path_py_bf)
    except nx.NetworkXNoPath:
        path_py_bf = None
        path_len_py_bf = 0
    t1 = time.time()

    # Bellman-Ford with C++ backend (includes conversion time)
    try:
        path_cpp_bf = nx.shortest_path(G, source, target, weight="weight", method="bellman-ford", backend="cpp")
        path_len_cpp_bf = len(path_cpp_bf)
    except nx.NetworkXNoPath:
        path_cpp_bf = None
        path_len_cpp_bf = 0
    t2 = time.time()

    print(f"NetworkX (py): {t1 - t0:.3f}s, path length: {path_len_py_bf}")
    print(f"nx-cpp backend (includes conversion): {t2 - t1:.3f}s, path length: {path_len_cpp_bf}")
    print(f"Speedup: {(t1 - t0)/(t2 - t1):.2f}x")
    same_length_bf = path_len_py_bf == path_len_cpp_bf
    print(f"\nVerification: Both paths have length {path_len_py_bf}: {same_length_bf}")

    print("\n" + "=" * 50)
    print("All Shortest Paths from Source")
    print("=" * 50)

    # Get all paths from source (no target specified)
    t0 = time.time()
    paths_py = nx.shortest_path(G, source, weight="weight", method="dijkstra")
    t1 = time.time()

    paths_cpp = nx.shortest_path(G, source, weight="weight", method="dijkstra", backend="cpp")
    t2 = time.time()

    print(f"NetworkX (py): {t1 - t0:.3f}s, {len(paths_py)} reachable nodes")
    print(f"nx-cpp backend (includes conversion): {t2 - t1:.3f}s, {len(paths_cpp)} reachable nodes")
    if (t2 - t1) > 0:
        print(f"Speedup: {(t1 - t0)/(t2 - t1):.2f}x")

    # Verify same number of reachable nodes
    print(f"\nVerification: Both found {len(paths_py)} reachable nodes: {len(paths_py) == len(paths_cpp)}")


if __name__ == "__main__":
    nx.config.warnings_to_ignore.add("cache")
    main()
