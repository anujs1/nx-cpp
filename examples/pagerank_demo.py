import time, networkx as nx


def main():
    print("=== PageRank Demo ===")

    print("Generating graph...")
    G = nx.gnp_random_graph(10_000, 0.05, directed=True)

    print("Running PageRank with NetworkX (py)...")
    t0 = time.time()
    _ = nx.pagerank(G, alpha=0.85)
    t1 = time.time()

    print("Running PageRank with nx-cpp backend...")
    _ = nx.pagerank(G, alpha=0.85, backend="cpp")
    t2 = time.time()

    print(f"NetworkX (py): {t1 - t0:.3f}s")
    print(f"nx-cpp backend: {t2 - t1:.3f}s")
    print(f"Speedup: {(t1 - t0)/(t2 - t1):.2f}x")


if __name__ == "__main__":
    main()
