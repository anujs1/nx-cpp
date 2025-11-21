import time
import networkx as nx
import pytest

def compare_pageranks(pr_py, pr_cpp, tol=1e-6, msg_prefix=""):
    assert set(pr_py.keys()) == set(pr_cpp.keys()), f"{msg_prefix} node sets differ"
    for node in pr_py:
        a, b = pr_py[node], pr_cpp[node]
        assert abs(a - b) < tol, (
            f"{msg_prefix} mismatch on node {node}: {a} vs {b}"
        )

# UNIT TESTS (CORRECTNESS)

@pytest.mark.unit
def test_pagerank_simple_triangle():
    G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    pr_py = nx.pagerank(G)
    pr_cpp = nx.pagerank(G, backend="cpp")
    compare_pageranks(pr_py, pr_cpp, msg_prefix="triangle:")

@pytest.mark.unit
def test_pagerank_disconnected_components():
    G = nx.DiGraph([(1, 2), (2, 1), (3, 4), (4, 3)])
    pr_py = nx.pagerank(G)
    pr_cpp = nx.pagerank(G, backend="cpp")
    compare_pageranks(pr_py, pr_cpp, msg_prefix="disconnected:")

@pytest.mark.unit
def test_pagerank_star_graph():
    G = nx.DiGraph()
    center = 0
    for i in range(1, 6):
        G.add_edge(center, i)

    pr_py = nx.pagerank(G)
    pr_cpp = nx.pagerank(G, backend="cpp")
    compare_pageranks(pr_py, pr_cpp, msg_prefix="star:")

@pytest.mark.unit
def test_pagerank_mixed_node_types():
    G = nx.DiGraph()
    G.add_edges_from([
        (0, "A"),
        ("A", 3.14),
        (3.14, (1, 2)),
        ((1, 2), 0)
    ])
    pr_py = nx.pagerank(G)
    pr_cpp = nx.pagerank(G, backend="cpp")
    compare_pageranks(pr_py, pr_cpp, msg_prefix="mixed_node_types:")

@pytest.mark.unit
def test_pagerank_random_small(rng_seed):
    G = nx.gnp_random_graph(200, 0.05, directed=True, seed=rng_seed)
    pr_py = nx.pagerank(G)
    pr_cpp = nx.pagerank(G, backend="cpp")
    compare_pageranks(pr_py, pr_cpp, msg_prefix="random_small:")

@pytest.mark.unit
def test_pagerank_random_large(rng_seed):
    G = nx.gnp_random_graph(5000, 0.02, directed=True, seed=rng_seed)
    pr_py = nx.pagerank(G)
    pr_cpp = nx.pagerank(G, backend="cpp")
    compare_pageranks(pr_py, pr_cpp, msg_prefix="random_large:")

# GRACEFUL FALLBACK TESTS

@pytest.mark.graceful_fallback
def test_pagerank_cpp_exception_falls_back(monkeypatch, rng_seed):
    """
    simulate a failure in the C++ backend and assert that pagerank falls back to the Python backend
    """
    import nx_cpp.backend as cpp_backend

    G = nx.gnp_random_graph(100, 0.05, directed=True, seed=rng_seed)
    pr_py = nx.pagerank(G)

    def fail(*args, **kwargs):
        raise RuntimeError("Simulated C++ error")

    # patch the imported C++ function inside the backend module
    monkeypatch.setattr(cpp_backend, "_cpp_pagerank", fail, raising=True)

    pr_cpp = nx.pagerank(G, backend="cpp")
    compare_pageranks(pr_py, pr_cpp, msg_prefix="fallback_cpp_exception:")

@pytest.mark.graceful_fallback
def test_pagerank_weight_falls_back():
    """
    if weight is not 'weight' or None, falls back to Python
    """
    G = nx.DiGraph()
    G.add_edge(0, 1, w=4)
    G.add_edge(1, 2, w=2)
    pr_py = nx.pagerank(G, weight="w")
    pr_cpp = nx.pagerank(G, weight="w", backend="cpp")
    compare_pageranks(pr_py, pr_cpp, msg_prefix="fallback_weight:")

@pytest.mark.graceful_fallback
def test_pagerank_personalization_falls_back():
    """
    personalization is unsupported -> must fall back
    """
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    p = {0: 0.5, 1: 0.3, 2: 0.2}
    pr_py = nx.pagerank(G, personalization=p)
    pr_cpp = nx.pagerank(G, personalization=p, backend="cpp")
    compare_pageranks(pr_py, pr_cpp, msg_prefix="fallback_personalization:")


@pytest.mark.graceful_fallback
def test_pagerank_nstart_falls_back():
    """
    nstart is unsupported -> must fall back
    """
    G = nx.DiGraph([(0, 1), (1, 2)])
    nstart = {0: 0.9, 1: 0.1, 2: 0.0}
    pr_py = nx.pagerank(G, nstart=nstart)
    pr_cpp = nx.pagerank(G, nstart=nstart, backend="cpp", )
    compare_pageranks(pr_py, pr_cpp, msg_prefix="fallback_nstart:")

@pytest.mark.graceful_fallback
def test_pagerank_dangling_falls_back():
    """
    dangling is unsupported -> must fall back
    """
    G = nx.DiGraph([(0, 1), (1, 2)])
    pr_py = nx.pagerank(G, dangling={0: 0.5, 1: 0.3, 2: 0.2})
    pr_cpp = nx.pagerank(G, dangling={0: 0.5, 1: 0.3, 2: 0.2}, backend="cpp")
    compare_pageranks(pr_py, pr_cpp, msg_prefix="fallback_dangling:")

# NYC ROAD TEST (CORRECTNESS & SPEEDUP)

@pytest.mark.nyc
@pytest.mark.performance
def test_pagerank_nyc_correctness_and_speedup(nyc_graph):
    G = nyc_graph
    # manually clearing cache from previous tests
    G.__networkx_cache__.clear()

    t0 = time.perf_counter()
    pr_py = nx.pagerank(G)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    pr_cpp = nx.pagerank(G, backend="cpp")
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.pagerank(G, backend="cpp")
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache

    compare_pageranks(pr_py, pr_cpp, msg_prefix="NYC:")

    assert abs(sum(pr_cpp.values()) - 1.0) < 1e-6

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        "[NYC PageRank]\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0

# USA-NORTHEAST ROAD TEST (SPEEDUP)

@pytest.mark.usa_ne
@pytest.mark.performance
def test_pagerank_usa_ne_speedup(usa_ne_graph):
    """
    focused on speedup since any correctness issues should be caught in NYC test
    """
    G = usa_ne_graph
    # manually clearing cache from previous tests
    G.__networkx_cache__.clear()

    t0 = time.perf_counter()
    _ = nx.pagerank(G)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.pagerank(G, backend="cpp")
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.pagerank(G, backend="cpp")
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")
    
    print("")
    print(
        "[USA North-East PageRank]\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0

# USA-EAST ROAD TEST (SPEEDUP)

@pytest.mark.usa_e
@pytest.mark.performance
@pytest.mark.slow
def test_pagerank_usa_e_speedup(usa_e_graph):
    """
    focused on speedup since any correctness issues should be caught in NYC test
    """
    G = usa_e_graph
    # manually clearing cache from previous tests
    G.__networkx_cache__.clear()

    t0 = time.perf_counter()
    _ = nx.pagerank(G)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.pagerank(G, backend="cpp")
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.pagerank(G, backend="cpp")
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        "[USA East PageRank]\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0