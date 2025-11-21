import math
import time

import networkx as nx
import pytest

def path_length(G, path, weight_attr="weight"):
    """
    Compute total path length given a list of nodes
    """
    if len(path) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(path, path[1:]):
        data = G[u][v]
        w = data.get(weight_attr, 1.0)
        total += w
    return total


def compare_shortest_path_weighted(G, path_py, path_cpp, msg_prefix=""):
    """
    Compare two weighted shortest paths:
    - same endpoints
    - same total path length (within float tolerance)
    """
    assert path_py[0] == path_cpp[0], f"{msg_prefix} start nodes differ"
    assert path_py[-1] == path_cpp[-1], f"{msg_prefix} end nodes differ"

    len_py = path_length(G, path_py, "weight")
    len_cpp = path_length(G, path_cpp, "weight")

    assert math.isclose(
        len_py, len_cpp, rel_tol=1e-9, abs_tol=1e-9
    ), (
        f"{msg_prefix} path lengths differ: "
        f"Python={len_py}, C++={len_cpp}\n"
        f"Python path: {path_py}\n"
        f"C++ path: {path_cpp}"
    )


def compare_shortest_path_unweighted(path_py, path_cpp, msg_prefix=""):
    """
    Compare two unweighted shortest paths:
    - same endpoints
    - same number of edges
    """
    assert path_py[0] == path_cpp[0], f"{msg_prefix} start nodes differ"
    assert path_py[-1] == path_cpp[-1], f"{msg_prefix} end nodes differ"

    edges_py = len(path_py)
    edges_cpp = len(path_cpp)

    assert edges_py == edges_cpp, (
        f"{msg_prefix} unweighted path lengths differ: "
        f"Python={edges_py}, C++={edges_cpp}\n"
        f"Python path: {path_py}\n"
        f"C++ path: {path_cpp}"
    )


def pick_reachable_target(G, source):
    """
    Given a source node, pick a reachable target by running a BFS tree and taking the last discovered node
    """
    bfs_nodes = [source] + [v for _, v in nx.bfs_edges(G, source)]
    return bfs_nodes[-1] if bfs_nodes else source


# UNIT TESTS (CORRECTNESS)

@pytest.mark.unit
def test_shortest_path_dijkstra_simple_weighted():
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        (0, 1, 1.0),
        (1, 2, 2.0),
        (0, 2, 5.0),
    ])
    source = 0
    target = 2

    py_path = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra")
    cpp_path = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra", backend="cpp")

    compare_shortest_path_weighted(G, py_path, cpp_path, msg_prefix="dijkstra_simple_weighted:")


@pytest.mark.unit
def test_shortest_path_bellman_ford_simple_weighted():
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        (0, 1, 1.0),
        (1, 2, -2.0),
        (0, 2, 0.5),
    ])
    source = 0
    target = 2

    py_path = nx.shortest_path(G, source=source, target=target, weight="weight", method="bellman-ford")
    cpp_path = nx.shortest_path(G, source=source, target=target, weight="weight", method="bellman-ford", backend="cpp")

    compare_shortest_path_weighted(G, py_path, cpp_path, msg_prefix="bellman_ford_simple_weighted:")


@pytest.mark.unit
def test_shortest_path_unweighted_simple():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 3)])
    source = 0
    target = 3

    py_path = nx.shortest_path(G, source=source, target=target, weight=None, method="dijkstra")
    cpp_path = nx.shortest_path(G, source=source, target=target, weight=None, method="dijkstra", backend="cpp")

    compare_shortest_path_unweighted(py_path, cpp_path, msg_prefix="unweighted_simple:")


@pytest.mark.unit
def test_shortest_path_mixed_node_types():
    G = nx.DiGraph()
    G.add_edges_from([
        (0, "A"),
        ("A", 3.14),
        (3.14, (1, 2)),
        ((1, 2), "Z"),
    ])

    source = 0
    target = "Z"

    py_path = nx.shortest_path(G, source=source, target=target, weight=None, method="dijkstra")
    cpp_path = nx.shortest_path(G, source=source, target=target, weight=None, method="dijkstra", backend="cpp")

    compare_shortest_path_unweighted(py_path, cpp_path, msg_prefix="mixed_node_types:")


@pytest.mark.unit
def test_shortest_path_random_small_weighted(rng_seed):
    G = nx.gnp_random_graph(200, 0.05, directed=True, seed=rng_seed)
    for u, v in G.edges():
        G[u][v]["weight"] = (hash((u, v, rng_seed)) % 10) + 1.0

    source = next(iter(G.nodes))
    target = pick_reachable_target(G, source)

    py_path = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra")
    cpp_path = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra", backend="cpp")

    compare_shortest_path_weighted(G, py_path, cpp_path, msg_prefix="random_small_weighted:")


@pytest.mark.unit
def test_shortest_path_random_small_unweighted(rng_seed):
    G = nx.gnp_random_graph(200, 0.05, directed=True, seed=rng_seed)
    source = next(iter(G.nodes))
    target = pick_reachable_target(G, source)

    py_path = nx.shortest_path(G, source=source, target=target, weight=None, method="dijkstra")
    cpp_path = nx.shortest_path(G, source=source, target=target, weight=None, method="dijkstra", backend="cpp")

    compare_shortest_path_unweighted(py_path, cpp_path, msg_prefix="random_small_unweighted:")


# GRACEFUL FALLBACK TESTS

@pytest.mark.graceful_fallback
def test_shortest_path_source_none_falls_back():
    """
    source=None is unsupported by the C++ backend -> must fall back.
    """
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        (0, 1, 1.0),
        (1, 2, 2.0),
        (0, 2, 10.0),
    ])

    py_result = nx.shortest_path(G, source=None, target=2, weight="weight", method="dijkstra")

    cpp_result = nx.shortest_path(G, source=None, target=2, weight="weight", method="dijkstra", backend="cpp")

    assert py_result == cpp_result, "source=None fallback mismatch"


@pytest.mark.graceful_fallback
def test_shortest_path_cpp_exception_falls_back(monkeypatch, rng_seed):
    """
    Simulate a failure in the C++ backend and assert that shortest_path falls back to Python
    """
    import nx_cpp.backend as backend

    G = nx.gnp_random_graph(100, 0.05, directed=True, seed=rng_seed)
    for u, v in G.edges():
        G[u][v]["weight"] = (hash((u, v, rng_seed)) % 5) + 1.0

    source = next(iter(G.nodes))
    target = pick_reachable_target(G, source)

    py_path = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra")

    def fail(*args, **kwargs):
        raise RuntimeError("Simulated C++ error")

    monkeypatch.setattr(backend, "_cpp_dijkstra", fail, raising=True)

    cpp_path = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra", backend="cpp")

    compare_shortest_path_weighted(G, py_path, cpp_path, msg_prefix="fallback_cpp_exception:")


# ROAD NETWORK TESTS (CORRECTNESS & PERFORMANCE)

@pytest.mark.nyc
@pytest.mark.performance
def test_shortest_path_nyc_correctness_and_speedup(nyc_graph):
    G = nyc_graph
    # manually clearing cache from previous tests
    G.__networkx_cache__.clear()

    source = next(iter(G.nodes))
    target = pick_reachable_target(G, source)

    t0 = time.perf_counter()
    py_path = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra")
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    cpp_path = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra", backend="cpp")
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra", backend="cpp")
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache
    compare_shortest_path_weighted(G, py_path, cpp_path, msg_prefix="NYC:")

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        f"[NYC shortest_path]\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0


@pytest.mark.usa_ne
@pytest.mark.performance
def test_shortest_path_usa_ne_speedup(usa_ne_graph):
    G = usa_ne_graph
    # manually clearing cache from previous tests
    G.__networkx_cache__.clear()

    source = next(iter(G.nodes))
    target = pick_reachable_target(G, source)

    t0 = time.perf_counter()
    py_path = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra")
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    cpp_path = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra", backend="cpp")
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra", backend="cpp")
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache
    compare_shortest_path_weighted(G, py_path, cpp_path, msg_prefix="USA-NE:")

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        f"[USA North-East shortest_path]\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0


@pytest.mark.usa_e
@pytest.mark.performance
@pytest.mark.slow
def test_shortest_path_usa_e_speedup(usa_e_graph):
    G = usa_e_graph
    # manually clearing cache from previous tests
    G.__networkx_cache__.clear()

    source = next(iter(G.nodes))
    target = pick_reachable_target(G, source)

    t0 = time.perf_counter()
    py_path = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra")
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    cpp_path = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra", backend="cpp")
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = nx.shortest_path(G, source=source, target=target, weight="weight", method="dijkstra", backend="cpp")
    t_cpp_cache = time.perf_counter() - t0

    conversion_overhead_estimate = t_cpp - t_cpp_cache
    compare_shortest_path_weighted(G, py_path, cpp_path, msg_prefix="USA-E:")

    speedup = t_py / t_cpp if t_cpp > 0 else float("inf")

    print("")
    print(
        f"[USA East shortest_path]\n"
        f"python={t_py:.3f}s cpp={t_cpp:.3f}s\n"
        f"est. cpp graph conversion time={conversion_overhead_estimate:.3f}s\n"
        f"est. algo time={t_cpp_cache:.3f}s\n"
        f"total speedup={speedup:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_cache):.2f}x"
    )

    assert speedup > 1.0