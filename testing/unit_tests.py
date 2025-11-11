import networkx as nx
import random


def compare_pagerank(G, tol=1e-6):
    pr_py = nx.pagerank(G, alpha=0.85)
    pr_cpp = nx.pagerank(G, alpha=0.85, backend="cpp")
    assert set(pr_py.keys()) == set(pr_cpp.keys()), "Node sets differ"
    for node in pr_py:
        assert abs(pr_py[node] - pr_cpp[node]) < tol, f"Mismatch on node {node}, {pr_py[node]} vs {pr_cpp[node]}"


def test_simple_triangle():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    compare_pagerank(G)


def test_disconnected_components():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 1), (3, 4), (4, 3)])
    compare_pagerank(G)


def test_star_graph():
    G = nx.DiGraph()
    center = 0
    for i in range(1, 6):
        G.add_edge(center, i)
    compare_pagerank(G)


def test_random_small():
    G = nx.gnp_random_graph(50, 0.1, directed=True, seed=random_int)
    compare_pagerank(G)


if __name__ == "__main__":
    random_int = random.randint(1, 10000)
    print("Random seed:", random_int)
    test_simple_triangle()
    print("Triangle test passed")
    test_disconnected_components()
    print("Disconnected components test passed")
    test_star_graph()
    print("Star graph test passed")
    test_random_small()
    print("All unit tests passed")
