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

# UNIT TESTS

@pytest.mark.unit
def test_pagerank_simple_triangle():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])

    pr_py = nx.pagerank(G, alpha=0.85)
    pr_cpp = nx.pagerank(G, alpha=0.85, backend="cpp")

    compare_pageranks(pr_py, pr_cpp, msg_prefix="triangle:")


@pytest.mark.unit
def test_pagerank_disconnected_components():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (2, 1), (3, 4), (4, 3)])

    pr_py = nx.pagerank(G, alpha=0.85)
    pr_cpp = nx.pagerank(G, alpha=0.85, backend="cpp")

    compare_pageranks(pr_py, pr_cpp, msg_prefix="disconnected:")


@pytest.mark.unit
def test_pagerank_star_graph():
    G = nx.DiGraph()
    center = 0
    for i in range(1, 6):
        G.add_edge(center, i)

    pr_py = nx.pagerank(G, alpha=0.85)
    pr_cpp = nx.pagerank(G, alpha=0.85, backend="cpp")

    compare_pageranks(pr_py, pr_cpp, msg_prefix="star:")


@pytest.mark.unit
def test_pagerank_random_small(rng_seed):
    G = nx.gnp_random_graph(50, 0.1, directed=True, seed=rng_seed)

    pr_py = nx.pagerank(G, alpha=0.85)
    pr_cpp = nx.pagerank(G, alpha=0.85, backend="cpp")

    compare_pageranks(pr_py, pr_cpp, msg_prefix="random_small:")

# GRACEFUL FALLBACK TESTS

# @pytest.mark.graceful_fallback
# def test_pagerank_cpp_exception_falls_back(monkeypatch):
#     """
#     Simulate a failure in the C++ backend and assert that
#     nx.pagerank(..., backend='cpp') falls back to the pure Python impl.

#     Contract:
#     - C++ failure must NOT crash user code
#     - Result must match Python pagerank
#     """
#     import nx_cpp.backend as cpp_backend

#     G = nx.gnp_random_graph(40, 0.15, directed=True, seed=123)
#     pr_py = nx.pagerank(G, alpha=0.85)

#     def boom(*args, **kwargs):
#         raise RuntimeError("Simulated C++ pagerank failure")

#     # Patch the imported C++ function inside the backend module
#     monkeypatch.setattr(cpp_backend, "_cpp_pagerank", boom, raising=True)

#     pr_cpp = nx.pagerank(G, alpha=0.85, backend="cpp")

#     compare_pageranks(pr_py, pr_cpp, msg_prefix="graceful_fallback:")


# NYC ROAD TEST (CORRECTNESS & SPEEDUP)

@pytest.mark.nyc
@pytest.mark.performance
def test_pagerank_nyc_correctness_and_timing(nyc_graph):
    G = nyc_graph

    t0 = time.perf_counter()
    pr_py = nx.pagerank(G, alpha=0.85)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    pr_cpp = nx.pagerank(G, alpha=0.85, backend="cpp")
    t_cpp = time.perf_counter() - t0

    compare_pageranks(pr_py, pr_cpp, tol=1e-5, msg_prefix="NYC:")

    assert abs(sum(pr_cpp.values()) - 1.0) < 1e-5

    if t_cpp > 0:
        speedup = t_py / t_cpp
    else:
        speedup = float("inf")

    print(
        f"[NYC PageRank] python={t_py:.3f}s cpp={t_cpp:.3f}s "
        f"speedup={speedup:.2f}x"
    )

    assert speedup > 1.0

# USA ROAD TEST (SPEEDUP)

@pytest.mark.usa
@pytest.mark.performance
def test_pagerank_usa_speedup(usa_graph):
    """
    Focused on speedup since any correctness issues should be caught in NYC test
    """
    G = usa_graph

    t0 = time.perf_counter()
    _ = nx.pagerank(G, alpha=0.85)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.pagerank(G, alpha=0.85, backend="cpp")
    t_cpp = time.perf_counter() - t0

    if t_cpp > 0:
        speedup = t_py / t_cpp
    else:
        speedup = float("inf")

    print(
        f"[USA PageRank] python={t_py:.3f}s cpp={t_cpp:.3f}s "
        f"speedup={speedup:.2f}x"
    )

    assert speedup > 1.0