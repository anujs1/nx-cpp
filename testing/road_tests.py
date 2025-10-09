import time
import networkx as nx
from road_loader import load_dimacs_graph

def test_pagerank_nyc_correctness(tol=1e-5):
    path = "USA-road-d.NY.gr"
    G = load_dimacs_graph(path, directed=True)

    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    start = time.perf_counter()
    pr_py = nx.pagerank(G, alpha=0.85)
    t_py = time.perf_counter() - start

    start = time.perf_counter()
    pr_cpp = nx.pagerank(G, alpha=0.85, backend='cpp')
    t_cpp = time.perf_counter() - start

    assert set(pr_py.keys()) == set(pr_cpp.keys()), "Node sets differ"
    for node in pr_py:
        assert abs(pr_py[node] - pr_cpp[node]) < tol, f"Mismatch on node {node}, {pr_py[node]} vs {pr_cpp[node]}"

    print(f"Python={t_py:.4f}s  C++={t_cpp:.4f}s  Speedup={t_py/t_cpp:.2f}x")

    assert abs(sum(pr_cpp.values()) - 1) < tol, "PageRank values do not sum to 1"

if __name__ == "__main__":
    test_pagerank_nyc_correctness()
    print("Road tests passed")