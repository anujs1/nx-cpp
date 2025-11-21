import time
import networkx as nx
import pytest

def compare_dfs_edges(edges_py, edges_cpp, msg_prefix=""):
    edges_py = list(edges_py)
    edges_cpp = list(edges_cpp)
    assert len(edges_py) == len(edges_cpp), f"{msg_prefix} edge counts differ"
    for i, (a, b) in enumerate(zip(edges_py, edges_cpp)):
        assert a == b, f"{msg_prefix} mismatch on edge {i}: {a} vs {b}"

# UNIT TESTS (CORRECTNESS)

@pytest.mark.unit
def test_dfs_simple_line():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    source = 0
    py_edges = nx.dfs_edges(G, source)
    cpp_edges = nx.dfs_edges(G, source, backend="cpp")
    compare_dfs_edges(py_edges, cpp_edges, msg_prefix="simple_line:")

@pytest.mark.unit
def test_dfs_simple_branching():
    G = nx.DiGraph([(0, 1), (0, 2), (2, 3)])
    source = 0
    py_edges = nx.dfs_edges(G, source)
    cpp_edges = nx.dfs_edges(G, source, backend="cpp")
    compare_dfs_edges(py_edges, cpp_edges, msg_prefix="simple_branching:")

@pytest.mark.unit
def test_dfs_disconnected_components():
    G = nx.DiGraph([(0, 1), (2, 3)])
    source = 0
    py_edges = nx.dfs_edges(G, source)
    cpp_edges = nx.dfs_edges(G, source, backend="cpp")
    compare_dfs_edges(py_edges, cpp_edges, msg_prefix="disconnected:")

@pytest.mark.unit
def test_dfs_undirected_graph():
    G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)])
    source = 0
    py_edges = nx.dfs_edges(G, source)
    cpp_edges = nx.dfs_edges(G, source, backend="cpp")
    compare_dfs_edges(py_edges, cpp_edges, msg_prefix="undirected:")

@pytest.mark.unit
def test_dfs_mixed_node_types():
    G = nx.DiGraph()
    G.add_edges_from([
        (0, "A"),
        ("A", 3.14),
        (3.14, (1, 2)),
        ((1, 2), 0),
    ])
    source = 0
    py_edges = nx.dfs_edges(G, source)
    cpp_edges = nx.dfs_edges(G, source, backend="cpp")
    compare_dfs_edges(py_edges, cpp_edges, msg_prefix="mixed_node_types:")

@pytest.mark.unit
def test_dfs_random_small(rng_seed):
    G = nx.gnp_random_graph(200, 0.05, directed=True, seed=rng_seed)
    source = next(iter(G.nodes))
    py_edges = nx.dfs_edges(G, source)
    cpp_edges = nx.dfs_edges(G, source, backend="cpp")
    compare_dfs_edges(py_edges, cpp_edges, msg_prefix="random_small:")

@pytest.mark.unit
def test_dfs_random_large(rng_seed):
    G = nx.gnp_random_graph(5000, 0.02, directed=True, seed=rng_seed)
    source = next(iter(G.nodes))
    py_edges = nx.dfs_edges(G, source)
    cpp_edges = nx.dfs_edges(G, source, backend="cpp")
    compare_dfs_edges(py_edges, cpp_edges, msg_prefix="random_large:")

# GRACEFUL FALLBACK TESTS

@pytest.mark.graceful_fallback
def test_dfs_cpp_exception_falls_back(monkeypatch, rng_seed):
    """
    simulate a failure in the C++ backend and assert that dfs_edges falls back to the Python backend
    """
    import nx_cpp.backend as backend

    G = nx.gnp_random_graph(100, 0.05, directed=True, seed=rng_seed)
    source = next(iter(G.nodes))

    py_edges = nx.dfs_edges(G, source)

    def fail(*args, **kwargs):
        raise RuntimeError("Simulated C++ error")

    monkeypatch.setattr(backend, "_cpp_dfs_edges", fail, raising=True)

    cpp_edges = nx.dfs_edges(G, source, backend="cpp")
    compare_dfs_edges(py_edges, cpp_edges, msg_prefix="fallback_cpp_exception:")

@pytest.mark.graceful_fallback
def test_dfs_depth_limit_falls_back():
    """
    depth_limit is unsupported -> must fall back
    """
    G = nx.balanced_tree(r=3, h=4, create_using=nx.DiGraph)
    source = 0
    py_edges = nx.dfs_edges(G, source, depth_limit=2)
    cpp_edges = nx.dfs_edges(G, source, depth_limit=2, backend="cpp")
    compare_dfs_edges(py_edges, cpp_edges, msg_prefix="fallback_depth_limit:")

@pytest.mark.graceful_fallback
def test_dfs_sort_neighbors_falls_back():
    """
    sort_neighbors is unsupported -> must fall back
    """
    G = nx.DiGraph([
        (0, 2),
        (0, 1),
        (0, 3),
        (1, 4),
        (1, 5),
    ])
    source = 0
    
    def sort_neighbors(nbrs):
        return sorted(nbrs)

    py_edges = nx.dfs_edges(G, source, sort_neighbors=sort_neighbors)
    cpp_edges = nx.dfs_edges(G, source, sort_neighbors=sort_neighbors, backend="cpp")
    compare_dfs_edges(py_edges, cpp_edges, msg_prefix="fallback_sort_neighbors:")

# NYC ROAD TEST (CORRECTNESS & SPEEDUP)

# typically fails --> highlights that graph conversion overhead makes cpp less useful for small graphs & fast functions
@pytest.mark.nyc
@pytest.mark.performance
def test_dfs_nyc_correctness_and_speedup(nyc_graph):
    G = nyc_graph
    # manually clearing cache from previous tests
    G.__networkx_cache__.clear()
    source = next(iter(G.nodes))

    t0 = time.perf_counter()
    py_edges = nx.dfs_edges(G, source)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    cpp_edges = nx.dfs_edges(G, source, backend="cpp")
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.dfs_edges(G, source, backend="cpp")
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache

    compare_dfs_edges(py_edges, cpp_edges, msg_prefix="NYC:")

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        f"[NYC DFS]\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0

# USA-NE (MEDIUM) — PERFORMANCE ONLY

# sometimes fails --> highlights that graph conversion overhead makes cpp less useful for small graphs & fast functions
@pytest.mark.usa_ne
@pytest.mark.performance
def test_dfs_usa_ne_speedup(usa_ne_graph):
    G = usa_ne_graph
    # manually clearing cache from previous tests
    G.__networkx_cache__.clear()
    source = next(iter(G.nodes))

    t0 = time.perf_counter()
    _ = list(nx.dfs_edges(G, source))
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = list(nx.dfs_edges(G, source, backend="cpp"))
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = list(nx.dfs_edges(G, source, backend="cpp"))
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        f"[USA North-East DFS]\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0

# USA-E (LARGE) — PERFORMANCE ONLY

@pytest.mark.usa_e
@pytest.mark.performance
@pytest.mark.slow
def test_dfs_usa_e_speedup(usa_e_graph):
    G = usa_e_graph
    # manually clearing cache from previous tests
    G.__networkx_cache__.clear()
    source = next(iter(G.nodes))

    t0 = time.perf_counter()
    _ = list(nx.dfs_edges(G, source))
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = list(nx.dfs_edges(G, source, backend="cpp"))
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = list(nx.dfs_edges(G, source, backend="cpp"))
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        f"[USA East DFS]\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0