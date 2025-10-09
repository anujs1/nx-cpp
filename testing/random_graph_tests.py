import networkx as nx
import time
import random

def test_random_graph_benchmark(n, p, tol=1e-6):
    G = nx.gnp_random_graph(n, p, directed=True, seed=random_int)

    start = time.perf_counter()
    pr_py = nx.pagerank(G, alpha=0.85)
    t_py = time.perf_counter() - start

    start = time.perf_counter()
    pr_cpp = nx.pagerank(G, alpha=0.85, backend="cpp")
    t_cpp = time.perf_counter() - start

    assert set(pr_py.keys()) == set(pr_cpp.keys()), "Node sets differ"
    for node in pr_py:
        assert abs(pr_py[node] - pr_cpp[node]) < tol, f"Mismatch on node {node}, {pr_py[node]} vs {pr_cpp[node]}"

    print(f"[n={n}, p={p}] Python={t_py:.4f}s  C++={t_cpp:.4f}s  Speedup={t_py/t_cpp:.2f}x")

if __name__ == "__main__":
    random_int = random.randint(1, 10000)
    print("Random seed:", random_int)
    test_random_graph_benchmark(1000, 0.01)
    test_random_graph_benchmark(2000, 0.005)
    test_random_graph_benchmark(10000, 0.1)
    test_random_graph_benchmark(20000, 0.05)
    print("All random graph tests passed")