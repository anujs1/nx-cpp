import time, networkx as nx


def main():
    G = nx.gnp_random_graph(10_000, 0.05, directed=True)

    t0 = time.time()
    _ = nx.pagerank(G, alpha=0.85)
    t1 = time.time()

    _ = nx.pagerank(G, alpha=0.85, backend="cpp")
    t2 = time.time()

    print(f"NetworkX (py): {t1 - t0:.3f}s")
    print(f"nx-cpp backend: {t2 - t1:.3f}s")
    if (t2 - t1) > 0:
        print(f"Speedup: {(t1 - t0)/(t2 - t1):.2f}x")


if __name__ == "__main__":
    main()
