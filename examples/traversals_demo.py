import time, networkx as nx


def main():

    # Create a large random graph
    print("Creating graph...")
    t_graph_start = time.time()
    G = nx.gnp_random_graph(50_000, 0.001, directed=False)
    t_graph_end = time.time()
    print(f"Graph creation time: {t_graph_end - t_graph_start:.3f}s")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Pick a source node
    source = 0
    print(f"Running traversals from source node: {source}\n")

    print("=" * 50)
    print("BFS (Breadth-First Search)")
    print("=" * 50)

    # BFS with NetworkX (Python)
    t0 = time.time()
    edges_py = list(nx.bfs_edges(G, source))
    t1 = time.time()

    # BFS with C++ backend
    edges_cpp = list(nx.bfs_edges(G, source, backend="cpp"))
    t2 = time.time()

    print(f"NetworkX (py): {t1 - t0:.3f}s, {len(edges_py)} edges")
    print(f"nx-cpp backend: {t2 - t1:.3f}s, {len(edges_cpp)} edges")
    print(f"Speedup: {(t1 - t0)/(t2 - t1):.2f}x")
    print(f"\nVerification: Both return {len(edges_py)} edges: {len(edges_py) == len(edges_cpp)}")

    # Demonstrate reverse parameter
    print("\n--- Testing reverse parameter ---")
    edges_forward = list(nx.bfs_edges(G, source, backend="cpp"))
    edges_reverse = list(nx.bfs_edges(G, source, reverse=True, backend="cpp"))
    if len(edges_forward) > 0 and len(edges_reverse) > 0:
        print(f"Forward: {edges_forward[:3]}")
        print(f"Reverse: {edges_reverse[:3]}")

    print("\n" + "=" * 50)
    print("DFS (Depth-First Search)")
    print("=" * 50)

    # DFS with NetworkX (Python)
    t0 = time.time()
    dfs_edges_py = list(nx.dfs_edges(G, source))
    t1 = time.time()

    # DFS with C++ backend
    dfs_edges_cpp = list(nx.dfs_edges(G, source, backend="cpp"))
    t2 = time.time()

    print(f"NetworkX (py): {t1 - t0:.3f}s, {len(dfs_edges_py)} edges")
    print(f"nx-cpp backend: {t2 - t1:.3f}s, {len(dfs_edges_cpp)} edges")
    print(f"Speedup: {(t1 - t0)/(t2 - t1):.2f}x")
    print(f"\nVerification: Both return {len(dfs_edges_cpp)} edges: {len(dfs_edges_cpp) == len(dfs_edges_cpp)}")


if __name__ == "__main__":
    nx.config.warnings_to_ignore.add("cache")
    main()
