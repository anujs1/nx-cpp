import networkx as nx
import random

import nx_cpp

random_int = random.randint(1, 10000)


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


def _component_sets(iterable):
    return {frozenset(comp) for comp in iterable}


def test_connected_components_algorithms():
    G = nx.Graph()
    # Component 1
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    # Component 2
    G.add_edges_from([(10, 11), (11, 12)])
    # Component 3 (isolated node)
    G.add_node(20)
    comps_py = _component_sets(nx.connected_components(G))
    comps_union = _component_sets(nx_cpp.connected_components_union_find(G))
    comps_label = _component_sets(nx_cpp.connected_components_bfs(G))
    assert comps_py == comps_union == comps_label


def _mst_weight(graph):
    return sum(data["weight"] for _, _, data in graph.edges(data=True))


def test_minimum_spanning_tree_variants():
    G = nx.Graph()
    edges = [
        ("a", "b", 4),
        ("a", "c", 2),
        ("b", "c", 1),
        ("b", "d", 5),
        ("c", "d", 8),
        ("c", "e", 10),
        ("d", "e", 2),
        ("d", "f", 6),
        ("e", "f", 3),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    mst_nx = nx.minimum_spanning_tree(G, weight="weight")
    mst_cpp_kruskal = nx_cpp.minimum_spanning_tree(G, algorithm="kruskal")
    mst_cpp_prim = nx_cpp.minimum_spanning_tree(G, algorithm="prim")
    expected_weight = _mst_weight(mst_nx)
    assert abs(_mst_weight(mst_cpp_kruskal) - expected_weight) < 1e-9
    assert abs(_mst_weight(mst_cpp_prim) - expected_weight) < 1e-9
    assert {frozenset(edge) for edge in mst_cpp_kruskal.edges()} == {
        frozenset(edge) for edge in mst_cpp_prim.edges()
    }


def test_graph_isomorphism_routines():
    G1 = nx.Graph()
    G1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])
    G2 = nx.Graph()
    mapping = {0: "a", 1: "b", 2: "c", 3: "d"}
    for u, v in G1.edges():
        G2.add_edge(mapping[u], mapping[v])
    assert nx_cpp.is_isomorphic(G1, G2)
    G3 = nx.Graph()
    G3.add_edges_from([(0, 1), (1, 2), (2, 0)])  # triangle only
    assert not nx_cpp.is_isomorphic(G1, G3)


if __name__ == "__main__":
    print("Random seed:", random_int)
    test_simple_triangle()
    print("Triangle test passed")
    test_disconnected_components()
    print("Disconnected components test passed")
    test_star_graph()
    print("Star graph test passed")
    test_random_small()
    print("Random small test passed")
    test_connected_components_algorithms()
    print("Connected components tests passed")
    test_minimum_spanning_tree_variants()
    print("MST tests passed")
    test_graph_isomorphism_routines()
    print("Graph isomorphism tests passed")
    print("All unit tests passed")