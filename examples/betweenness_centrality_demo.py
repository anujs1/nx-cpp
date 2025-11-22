import time, networkx as nx


def main():

    # Create a large random graph
    print("Creating graph...")
    t_graph_start = time.time()
    G = nx.gnp_random_graph(5_000, 0.005, directed=False)
    t_graph_end = time.time()
    print(f"Graph creation time: {t_graph_end - t_graph_start:.3f}s")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\n" + "=" * 50)
    print("Betweenness Centrality")
    print("=" * 50)

    # Betweenness centrality with NetworkX (Python)
    print("\nRunning NetworkX (Python) betweenness centrality...")
    t0 = time.time()
    bc_py = nx.betweenness_centrality(G, normalized=True)
    t1 = time.time()

    # Betweenness centrality with C++ backend
    print("Running nx-cpp backend betweenness centrality...")
    bc_cpp = nx.betweenness_centrality(G, normalized=True, backend="cpp")
    t2 = time.time()

    print(f"\nNetworkX (py): {t1 - t0:.3f}s")
    print(f"nx-cpp backend: {t2 - t1:.3f}s")
    print(f"Speedup: {(t1 - t0)/(t2 - t1):.2f}x")

    # Verify results are similar (allowing for small floating point differences)
    print("\n--- Verification ---")
    top_5_py = sorted(bc_py.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_cpp = sorted(bc_cpp.items(), key=lambda x: x[1], reverse=True)[:5]

    print("Top 5 nodes by betweenness (NetworkX):")
    for node, bc in top_5_py:
        print(f"  Node {node}: {bc:.6f}")

    print("\nTop 5 nodes by betweenness (nx-cpp):")
    for node, bc in top_5_cpp:
        print(f"  Node {node}: {bc:.6f}")

    # Check if results are close
    max_diff = max(abs(bc_py[node] - bc_cpp[node]) for node in bc_py.keys())
    print(f"\nMaximum difference: {max_diff:.2e}")
    print(f"Results match: {max_diff < 1e-6}")

    # Test with normalized=False
    print("\n" + "=" * 50)
    print("Betweenness Centrality (unnormalized)")
    print("=" * 50)

    t0 = time.time()
    bc_py_unnorm = nx.betweenness_centrality(G, normalized=False)
    t1 = time.time()

    bc_cpp_unnorm = nx.betweenness_centrality(G, normalized=False, backend="cpp")
    t2 = time.time()

    print(f"NetworkX (py): {t1 - t0:.3f}s")
    print(f"nx-cpp backend: {t2 - t1:.3f}s")
    print(f"Speedup: {(t1 - t0)/(t2 - t1):.2f}x")

    # Verify unnormalized results
    print("\n--- Verification (unnormalized) ---")
    top_5_py_unnorm = sorted(bc_py_unnorm.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_cpp_unnorm = sorted(bc_cpp_unnorm.items(), key=lambda x: x[1], reverse=True)[:5]

    print("Top 5 nodes by betweenness (NetworkX, unnormalized):")
    for node, bc in top_5_py_unnorm:
        print(f"  Node {node}: {bc:.2f}")

    print("\nTop 5 nodes by betweenness (nx-cpp, unnormalized):")
    for node, bc in top_5_cpp_unnorm:
        print(f"  Node {node}: {bc:.2f}")

    max_diff_unnorm = max(abs(bc_py_unnorm[node] - bc_cpp_unnorm[node]) for node in bc_py_unnorm.keys())
    print(f"\nMaximum difference (unnormalized): {max_diff_unnorm:.2e}")
    print(f"Results match: {max_diff_unnorm < 1e-6}")


if __name__ == "__main__":
    nx.config.warnings_to_ignore.add("cache")
    main()
