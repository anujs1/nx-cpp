import time
import networkx as nx
import pytest

def mst_edge_set(T):
    edges = set()
    for u, v in T.edges:
        if u == v:
            continue
        edges.add(frozenset((u, v)))
    return edges

def compare_mst(T_py, T_cpp, msg_prefix=""):
    e_py = mst_edge_set(T_py)
    e_cpp = mst_edge_set(T_cpp)
    assert e_py == e_cpp, f"{msg_prefix} MST edges differ: {e_py} vs {e_cpp}"


# UNIT TESTS (CORRECTNESS)

@pytest.mark.unit
def test_mst_simple_triangle():
    G = nx.Graph()
    G.add_edge(0, 1, weight=1)
    G.add_edge(1, 2, weight=2)
    G.add_edge(0, 2, weight=10)

    T_py = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")
    T_cpp = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal", backend="cpp")

    compare_mst(T_py, T_cpp, msg_prefix="simple_triangle_kruskal:")

    T_py_prim = nx.minimum_spanning_tree(G, weight="weight", algorithm="prim")
    T_cpp_prim = nx.minimum_spanning_tree(G, weight="weight", algorithm="prim", backend="cpp")

    compare_mst(T_py_prim, T_cpp_prim, msg_prefix="simple_triangle_prim:")


@pytest.mark.unit
def test_mst_mixed_node_types():
    G = nx.Graph()
    G.add_edge(0, "A", weight=1.5)
    G.add_edge("A", 3.14, weight=2.5)
    G.add_edge(3.14, (1, 2), weight=0.5)
    G.add_edge((1, 2), 0, weight=3.0)

    T_py = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")
    T_cpp = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal", backend="cpp")
    compare_mst(T_py, T_cpp, msg_prefix="mixed_node_types_kruskal:")


@pytest.mark.unit
def test_mst_random_small(rng_seed):
    G = nx.gnp_random_graph(100, 0.05, seed=rng_seed)
    for idx, (u, v) in enumerate(G.edges()):
        G[u][v]["weight"] = float(idx + 1)

    T_py = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")
    T_cpp = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal", backend="cpp")
    compare_mst(T_py, T_cpp, msg_prefix="random_small_kruskal:")

    T_py_prim = nx.minimum_spanning_tree(G, weight="weight", algorithm="prim")
    T_cpp_prim = nx.minimum_spanning_tree(G, weight="weight", algorithm="prim", backend="cpp")
    compare_mst(T_py_prim, T_cpp_prim, msg_prefix="random_small_prim:")


# GRACEFUL FALLBACK TESTS

@pytest.mark.graceful_fallback
def test_mst_cpp_exception_falls_back(monkeypatch, rng_seed):
    """
    simulate a failure in the C++ backend and assert that minimum_spanning_tree falls back to Python
    """
    import nx_cpp.backend as backend

    G = nx.gnp_random_graph(80, 0.1, seed=rng_seed)
    for u, v in G.edges:
        G[u][v]["weight"] = 1.0

    T_py = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")

    def fail(*args, **kwargs):
        raise RuntimeError("simulated C++ error")

    if hasattr(backend, "_cpp_minimum_spanning_tree"):
        monkeypatch.setattr(backend, "_cpp_minimum_spanning_tree", fail, raising=True)

    T_cpp = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal", backend="cpp")
    compare_mst(T_py, T_cpp, msg_prefix="fallback_cpp_exception:")


@pytest.mark.graceful_fallback
def test_mst_unknown_algorithm_falls_back():
    """
    algorithm='boruvka' is unsupported in C++ -> must fall back
    """
    G = nx.gnp_random_graph(40, 0.2, seed=123)
    for u, v in G.edges:
        G[u][v]["weight"] = 1.0

    T_py = nx.minimum_spanning_tree(G, weight="weight", algorithm="boruvka")
    T_cpp = nx.minimum_spanning_tree(G, weight="weight", algorithm="boruvka", backend="cpp")
    compare_mst(T_py, T_cpp, msg_prefix="fallback_boruvka:")


# PERFORMANCE TESTS ON RANDOM UNDIRECTED GRAPHS

def make_weighted_gnp(n, p, rng_seed, weight_key="weight"):
    G = nx.gnp_random_graph(n, p, seed=rng_seed)
    for idx, (u, v) in enumerate(G.edges()):
        G[u][v][weight_key] = float(idx + 1)
    return G


@pytest.mark.performance
def test_mst_random_medium_kruskal_correctness_and_speedup(rng_seed):
    n = 5000
    p = 0.002
    weight_key = "weight"

    G = make_weighted_gnp(n, p, rng_seed, weight_key=weight_key)

    t0 = time.perf_counter()
    T_py = nx.minimum_spanning_tree(G, weight=weight_key, algorithm="kruskal")
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    T_cpp = nx.minimum_spanning_tree(G, weight=weight_key, algorithm="kruskal", backend="cpp")
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.minimum_spanning_tree(G, weight=weight_key, algorithm="kruskal", backend="cpp")
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache

    compare_mst(T_py, T_cpp, msg_prefix="random_medium_kruskal:")

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        f"[Random medium MST (kruskal)]\n"
        f"n={n}, p={p}\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0


@pytest.mark.performance
def test_mst_random_medium_prim_correctness_and_speedup(rng_seed):
    n = 5000
    p = 0.002
    weight_key = "weight"

    G = make_weighted_gnp(n, p, rng_seed, weight_key=weight_key)

    t0 = time.perf_counter()
    T_py = nx.minimum_spanning_tree(G, weight=weight_key, algorithm="prim")
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    T_cpp = nx.minimum_spanning_tree(G, weight=weight_key, algorithm="prim", backend="cpp")
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.minimum_spanning_tree(G, weight=weight_key, algorithm="prim", backend="cpp")
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache

    compare_mst(T_py, T_cpp, msg_prefix="random_medium_prim:")

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        f"[Random medium MST (prim)]\n"
        f"n={n}, p={p}\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0


@pytest.mark.performance
@pytest.mark.slow
def test_mst_random_large_kruskal_speedup(rng_seed):
    n = 20000
    p = 0.0005
    weight_key = "weight"

    G = make_weighted_gnp(n, p, rng_seed, weight_key=weight_key)

    t0 = time.perf_counter()
    _ = nx.minimum_spanning_tree(G, weight=weight_key, algorithm="kruskal")
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.minimum_spanning_tree(G, weight=weight_key, algorithm="kruskal", backend="cpp")
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.minimum_spanning_tree(G, weight=weight_key, algorithm="kruskal", backend="cpp")
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache
    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        f"[Random large MST (kruskal)]\n"
        f"n={n}, p={p}\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0


@pytest.mark.performance
@pytest.mark.slow
def test_mst_random_large_prim_speedup(rng_seed):
    n = 20000
    p = 0.0005
    weight_key = "weight"

    G = make_weighted_gnp(n, p, rng_seed, weight_key=weight_key)

    t0 = time.perf_counter()
    _ = nx.minimum_spanning_tree(G, weight=weight_key, algorithm="prim")
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.minimum_spanning_tree(
        G, weight=weight_key, algorithm="prim", backend="cpp"
    )
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.minimum_spanning_tree(
        G, weight=weight_key, algorithm="prim", backend="cpp"
    )
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache
    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        f"[Random large MST (prim)]\n"
        f"n={n}, p={p}\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0