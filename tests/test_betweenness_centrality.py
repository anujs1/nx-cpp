import time
import networkx as nx
import pytest

def compare_betweenness(bc_py, bc_cpp, tol=1e-6, msg_prefix=""):
    assert set(bc_py.keys()) == set(bc_cpp.keys()), f"{msg_prefix} node sets differ"
    for node in bc_py:
        a, b = bc_py[node], bc_cpp[node]
        assert abs(a - b) < tol, f"{msg_prefix} mismatch on node {node}: {a} vs {b}"

# UNIT TESTS (CORRECTNESS)

@pytest.mark.unit
def test_betweenness_simple_path():
    G = nx.path_graph(4)
    bc_py = nx.betweenness_centrality(G)
    bc_cpp = nx.betweenness_centrality(G, backend="cpp")
    compare_betweenness(bc_py, bc_cpp, msg_prefix="simple_path:")

@pytest.mark.unit
def test_betweenness_cycle():
    G = nx.cycle_graph(5)
    bc_py = nx.betweenness_centrality(G)
    bc_cpp = nx.betweenness_centrality(G, backend="cpp")
    compare_betweenness(bc_py, bc_cpp, msg_prefix="cycle:")

@pytest.mark.unit
def test_betweenness_mixed_node_types():
    G = nx.Graph()
    G.add_edges_from([
        (0, "A"),
        ("A", 3.14),
        (3.14, (1, 2)),
        ((1, 2), 0),
    ])
    bc_py = nx.betweenness_centrality(G)
    bc_cpp = nx.betweenness_centrality(G, backend="cpp")
    compare_betweenness(bc_py, bc_cpp, msg_prefix="mixed_node_types:")

@pytest.mark.unit
def test_betweenness_random_small(rng_seed):
    G = nx.gnp_random_graph(100, 0.05, seed=rng_seed)
    bc_py = nx.betweenness_centrality(G)
    bc_cpp = nx.betweenness_centrality(G, backend="cpp")
    compare_betweenness(bc_py, bc_cpp, msg_prefix="random_small:")

@pytest.mark.unit
def test_betweenness_disconnected():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    G.add_edges_from([(10, 11), (11, 12), (12, 10)])

    bc_py = nx.betweenness_centrality(G)
    bc_cpp = nx.betweenness_centrality(G, backend="cpp")
    compare_betweenness(bc_py, bc_cpp, msg_prefix="disconnected:")

# GRACEFUL FALLBACK TESTS

@pytest.mark.graceful_fallback
def test_betweenness_directed_simple():
    G = nx.DiGraph()
    G.add_edges_from([
        (0, 1),
        (1, 2),
        (0, 2),
        (2, 3),
    ])

    bc_py = nx.betweenness_centrality(G, normalized=True, weight='weight')
    bc_cpp = nx.betweenness_centrality(G, normalized=True, weight='weight', backend="cpp")
    compare_betweenness(bc_py, bc_cpp, msg_prefix="directed_simple:")

@pytest.mark.graceful_fallback
def test_betweenness_cpp_exception_falls_back(monkeypatch, rng_seed):
    """
    simulate a failure in the C++ backend and assert that betweenness_centrality falls back to the Python backend
    """
    import nx_cpp.backend as backend

    G = nx.gnp_random_graph(80, 0.1, seed=rng_seed)
    bc_py = nx.betweenness_centrality(G)

    def fail(*args, **kwargs):
        raise RuntimeError("simulated C++ error")

    if hasattr(backend, "_cpp_betweenness_centrality"):
        monkeypatch.setattr(backend, "_cpp_betweenness_centrality", fail, raising=True)

    bc_cpp = nx.betweenness_centrality(G, backend="cpp")
    compare_betweenness(bc_py, bc_cpp, msg_prefix="fallback_cpp_exception:")


@pytest.mark.graceful_fallback
def test_betweenness_weight_and_k_fall_back(rng_seed):
    """
    weight and k are unsupported in the C++ implementation -> must fall back
    """
    G = nx.path_graph(6)
    for u, v in G.edges:
        G[u][v]["w"] = 1.0

    bc_py = nx.betweenness_centrality(G, weight="w", k=3, seed=rng_seed)
    bc_cpp = nx.betweenness_centrality(G, weight="w", k=3, seed=rng_seed, backend="cpp")
    compare_betweenness(bc_py, bc_cpp, msg_prefix="fallback_weight_k:")

# ROME ROAD TEST (CORRECTNESS & SPEEDUP) --> larger roads are too heavy for betweenness

@pytest.mark.rome
@pytest.mark.performance
def test_betweenness_rome_correctness_and_speedup(rome_graph):
    G = rome_graph
    # manually clearing cache from previous tests
    G.__networkx_cache__.clear()

    t0 = time.perf_counter()
    bc_py = nx.betweenness_centrality(G)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    bc_cpp = nx.betweenness_centrality(G, backend="cpp")
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.betweenness_centrality(G, backend="cpp")
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache

    compare_betweenness(bc_py, bc_cpp, msg_prefix="NYC:")

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        f"[Rome betweenness_centrality]\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0