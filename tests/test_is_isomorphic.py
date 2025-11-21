import time
import networkx as nx
import pytest

def compare_isomorphic(py_result, cpp_result, msg_prefix=""):
    assert py_result == cpp_result, f"{msg_prefix} isomorphism result differs: {py_result} vs {cpp_result}"


# UNIT TESTS (CORRECTNESS)

@pytest.mark.unit
def test_isomorphic_identical_graphs():
    G1 = nx.path_graph(5)
    G2 = nx.path_graph(5)

    iso_py = nx.is_isomorphic(G1, G2)
    iso_cpp = nx.is_isomorphic(G1, G2, backend="cpp")
    compare_isomorphic(iso_py, iso_cpp, msg_prefix="identical:")


@pytest.mark.unit
def test_non_isomorphic_graphs():
    G1 = nx.path_graph(4)
    G2 = nx.cycle_graph(4)

    iso_py = nx.is_isomorphic(G1, G2)
    iso_cpp = nx.is_isomorphic(G1, G2, backend="cpp")
    compare_isomorphic(iso_py, iso_cpp, msg_prefix="non_isomorphic:")


@pytest.mark.unit
def test_isomorphic_mixed_node_types():
    G1 = nx.Graph()
    G1.add_edges_from([
        (0, "A"),
        ("A", 3.14),
        (3.14, (1, 2)),
    ])

    mapping = {0: "X", "A": 42, 3.14: "Y", (1, 2): (9, 9)}
    G2 = nx.relabel_nodes(G1, mapping)

    iso_py = nx.is_isomorphic(G1, G2)
    iso_cpp = nx.is_isomorphic(G1, G2, backend="cpp")
    compare_isomorphic(iso_py, iso_cpp, msg_prefix="mixed_node_types:")


@pytest.mark.unit
def test_isomorphic_random_small(rng_seed):
    G1 = nx.gnp_random_graph(30, 0.2, seed=rng_seed)
    nodes = list(G1.nodes)
    mapping = {u: v for u, v in zip(nodes, reversed(nodes))}
    G2 = nx.relabel_nodes(G1, mapping)

    iso_py = nx.is_isomorphic(G1, G2)
    iso_cpp = nx.is_isomorphic(G1, G2, backend="cpp")
    compare_isomorphic(iso_py, iso_cpp, msg_prefix="random_small:")


# GRACEFUL FALLBACK TESTS

@pytest.mark.graceful_fallback
def test_is_isomorphic_cpp_exception_falls_back(monkeypatch, rng_seed):
    """
    simulate a failure in the C++ backend and assert that is_isomorphic falls back to Python
    """
    import nx_cpp.backend as backend

    G1 = nx.gnp_random_graph(20, 0.3, seed=rng_seed)
    G2 = G1.copy()

    iso_py = nx.is_isomorphic(G1, G2)

    def fail(*args, **kwargs):
        raise RuntimeError("simulated C++ error")

    if hasattr(backend, "_cpp_graphs_are_isomorphic"):
        monkeypatch.setattr(backend, "_cpp_graphs_are_isomorphic", fail, raising=True)

    iso_cpp = nx.is_isomorphic(G1, G2, backend="cpp")
    compare_isomorphic(iso_py, iso_cpp, msg_prefix="fallback_cpp_exception:")

# RANDOM MEDIUM GRAPH (PERFORMANCE)

@pytest.mark.performance
def test_is_isomorphic_random_medium_speedup(rng_seed):
    n = 1000
    p = 0.05
    G1 = nx.gnp_random_graph(n, p, seed=rng_seed)

    nodes = list(G1.nodes())
    permuted = list(reversed(nodes))
    mapping = {u: v for u, v in zip(nodes, permuted)}
    G2 = nx.relabel_nodes(G1, mapping)

    t0 = time.perf_counter()
    iso_py = nx.is_isomorphic(G1, G2)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    iso_cpp = nx.is_isomorphic(G1, G2, backend="cpp")
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.is_isomorphic(G1, G2, backend="cpp")
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache
    compare_isomorphic(iso_py, iso_cpp, msg_prefix="random_medium:")

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        f"[Random medium is_isomorphic]\n"
        f"n={n}, p={p}\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0

# NYC SUBGRAPH TEST (G_sub vs G_sub.copy(), CORRECTNESS & SPEEDUP)

def induced_subgraph_first_n(G, n_nodes):
    """
    take the induced subgraph on the first n_nodes in G.nodes()
    """
    nodes = []
    for i, u in enumerate(G.nodes()):
        if i >= n_nodes:
            break
        nodes.append(u)
    return G.subgraph(nodes).copy()


@pytest.mark.nyc
@pytest.mark.performance
def test_is_isomorphic_nyc_subgraph_correctness_and_speedup(nyc_graph):
    """
    Use a subgraph of the NYC road network instead of the full graph
    """
    sub_n = 3000
    G = nyc_graph
    # manually clearing cache from previous tests
    G.__networkx_cache__.clear()
    G_sub = induced_subgraph_first_n(G, sub_n)
    G1 = G_sub
    G2 = G_sub.copy()

    t0 = time.perf_counter()
    iso_py = nx.is_isomorphic(G1, G2)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    iso_cpp = nx.is_isomorphic(G1, G2, backend="cpp")
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.is_isomorphic(G1, G2, backend="cpp")
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache

    compare_isomorphic(iso_py, iso_cpp, msg_prefix="NYC_subgraph:")

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        f"[NYC subgraph (n={sub_n}) is_isomorphic]\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0