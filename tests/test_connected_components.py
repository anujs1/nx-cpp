import time
import networkx as nx
import pytest

def normalize_components(components):
    return {frozenset(c) for c in components}


def compare_connected_components(comps_py, comps_cpp, msg_prefix=""):
    """
    Compare two collections of connected components for equality
    Order of components and order of nodes within components is irrelevant
    """
    norm_py = normalize_components(comps_py)
    norm_cpp = normalize_components(comps_cpp)
    assert norm_py == norm_cpp, (
        f"{msg_prefix} connected components differ:\n"
        f"Python: {norm_py}\n"
        f"C++: {norm_cpp}"
    )

# UNIT TESTS (CORRECTNESS)

@pytest.mark.unit
def test_connected_components_simple_path():
    G = nx.path_graph(5)

    comps_py = list(nx.connected_components(G))

    comps_cpp_union = list(nx.connected_components(G, backend="cpp", method="union-find"))
    comps_cpp_bfs = list(nx.connected_components(G, backend="cpp", method="bfs"))

    compare_connected_components(comps_py, comps_cpp_union, msg_prefix="simple_path_union_find:")
    compare_connected_components(comps_py, comps_cpp_bfs, msg_prefix="simple_path_bfs:")


@pytest.mark.unit
def test_connected_components_disconnected():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    G.add_edges_from([(10, 11), (11, 12)])

    comps_py = list(nx.connected_components(G))

    comps_cpp_union = list(nx.connected_components(G, backend="cpp", method="union-find"))
    comps_cpp_bfs = list(nx.connected_components(G, backend="cpp", method="bfs"))

    compare_connected_components(comps_py, comps_cpp_union, msg_prefix="disconnected_union_find:")
    compare_connected_components(comps_py, comps_cpp_bfs, msg_prefix="disconnected_bfs:")


@pytest.mark.unit
def test_connected_components_mixed_node_types():
    G = nx.Graph()
    G.add_edges_from([
        (0, "A"),
        ("A", 3.14),
        (3.14, (1, 2)),
        ((1, 2), "Z"),
    ])

    comps_py = list(nx.connected_components(G))

    comps_cpp_union = list(nx.connected_components(G, backend="cpp", method="union-find"))
    comps_cpp_bfs = list(nx.connected_components(G, backend="cpp", method="bfs"))

    compare_connected_components(comps_py, comps_cpp_union, msg_prefix="mixed_node_types_union_find:")
    compare_connected_components(comps_py, comps_cpp_bfs, msg_prefix="mixed_node_types_bfs:")


@pytest.mark.unit
def test_connected_components_random_small(rng_seed):
    G = nx.gnp_random_graph(300, 0.02, seed=rng_seed)

    comps_py = list(nx.connected_components(G))

    comps_cpp_union = list(nx.connected_components(G, backend="cpp", method="union-find"))
    comps_cpp_bfs = list(nx.connected_components(G, backend="cpp", method="bfs"))

    compare_connected_components(comps_py, comps_cpp_union, msg_prefix="random_small_union_find:")
    compare_connected_components(comps_py, comps_cpp_bfs, msg_prefix="random_small_bfs:")


# GRACEFUL FALLBACK TESTS

@pytest.mark.graceful_fallback
def test_connected_components_cpp_exception_falls_back(monkeypatch, rng_seed):
    import nx_cpp.backend as backend

    G = nx.gnp_random_graph(150, 0.05, seed=rng_seed)
    comps_py = list(nx.connected_components(G))

    def fail(*args, **kwargs):
        raise RuntimeError("Simulated C++ error")

    monkeypatch.setattr(
        backend, "_cpp_connected_components_union_find", fail, raising=True
    )

    comps_cpp_union = list(nx.connected_components(G, backend="cpp", method="union-find"))
    compare_connected_components(comps_py, comps_cpp_union, msg_prefix="fallback_cpp_exception_union_find:")


# PERFORMANCE TESTS

def make_multi_component_path_graph(num_components, component_size):
    G = nx.Graph()
    node_id = 0
    for _ in range(num_components):
        nodes = list(range(node_id, node_id + component_size))
        edges = [(nodes[i], nodes[i + 1]) for i in range(component_size - 1)]

        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        node_id += component_size

    return G

@pytest.mark.performance
def test_connected_components_medium_multi_component_speedup():
    num_components = 1000
    component_size = 1000
    G = make_multi_component_path_graph(num_components, component_size)

    t0 = time.perf_counter()
    comps_py = list(nx.connected_components(G))
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    comps_cpp_union = list(nx.connected_components(G, backend="cpp", method="union-find"))
    t_cpp_union = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = list(nx.connected_components(G, backend="cpp", method="union-find"))
    t_cpp_union_cache = time.perf_counter() - t0

    conversion_union_estimate = t_cpp_union - t_cpp_union_cache

    compare_connected_components(comps_py, comps_cpp_union, msg_prefix="medium_union_find:")

    speedup_union = t_py / t_cpp_union if t_cpp_union > 0 else float("inf")

    print("")
    print(
        f"[Medium (~1M nodes) connected_components - union-find]\n"
        f"python={t_py:.3f}s cpp={t_cpp_union:.3f}s\n"
        f"est. cpp graph conversion time={conversion_union_estimate:.3f}s\n"
        f"est. algo time={t_cpp_union_cache:.3f}s\n"
        f"total speedup={speedup_union:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_union_cache):.2f}x"
    )

    # manually clearing cache from previous tests
    G.__networkx_cache__.clear()

    t0 = time.perf_counter()
    comps_cpp_bfs = list(nx.connected_components(G, backend="cpp", method="bfs"))
    t_cpp_bfs = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = list(nx.connected_components(G, backend="cpp", method="bfs"))
    t_cpp_bfs_cache = time.perf_counter() - t0

    conversion_bfs_estimate = t_cpp_bfs - t_cpp_bfs_cache

    compare_connected_components(comps_py, comps_cpp_bfs, msg_prefix="medium_bfs:")

    speedup_bfs = t_py / t_cpp_bfs if t_cpp_bfs > 0 else float("inf")

    print("")
    print(
        f"[Medium (~1M nodes) connected_components - bfs]\n"
        f"python={t_py:.3f}s cpp={t_cpp_bfs:.3f}s\n"
        f"est. cpp graph conversion time={conversion_bfs_estimate:.3f}s\n"
        f"est. algo time={t_cpp_bfs_cache:.3f}s\n"
        f"total speedup={speedup_bfs:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_bfs_cache):.2f}x"
    )

    assert speedup_union > 1.0 or speedup_bfs > 1.0

@pytest.mark.performance
@pytest.mark.slow
def test_connected_components_large_multi_component_speedup():
    num_components = 1000
    component_size = 3000
    G = make_multi_component_path_graph(num_components, component_size)

    t0 = time.perf_counter()
    comps_py = list(nx.connected_components(G))
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    comps_cpp_union = list(nx.connected_components(G, backend="cpp", method="union-find"))
    t_cpp_union = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = list(nx.connected_components(G, backend="cpp", method="union-find"))
    t_cpp_union_cache = time.perf_counter() - t0

    conversion_union_estimate = t_cpp_union - t_cpp_union_cache

    compare_connected_components(comps_py, comps_cpp_union, msg_prefix="large_union_find:")

    speedup_union = t_py / t_cpp_union if t_cpp_union > 0 else float("inf")

    print("")
    print(
        f"[Large (~3M nodes) connected_components - union-find]\n"
        f"python={t_py:.3f}s cpp={t_cpp_union:.3f}s\n"
        f"est. cpp graph conversion time={conversion_union_estimate:.3f}s\n"
        f"est. algo time={t_cpp_union_cache:.3f}s\n"
        f"total speedup={speedup_union:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_union_cache):.2f}x"
    )

    # manually clearing cache from previous tests
    G.__networkx_cache__.clear()
    t0 = time.perf_counter()
    comps_cpp_bfs = list(nx.connected_components(G, backend="cpp", method="bfs"))
    t_cpp_bfs = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = list(nx.connected_components(G, backend="cpp", method="bfs"))
    t_cpp_bfs_cache = time.perf_counter() - t0

    conversion_bfs_estimate = t_cpp_bfs - t_cpp_bfs_cache

    compare_connected_components(comps_py, comps_cpp_bfs, msg_prefix="large_bfs:")

    speedup_bfs = t_py / t_cpp_bfs if t_cpp_bfs > 0 else float("inf")

    print("")
    print(
        f"[Large (~3M nodes) connected_components - bfs]\n"
        f"python={t_py:.3f}s cpp={t_cpp_bfs:.3f}s\n"
        f"est. cpp graph conversion time={conversion_bfs_estimate:.3f}s\n"
        f"est. algo time={t_cpp_bfs_cache:.3f}s\n"
        f"total speedup={speedup_bfs:.2f}x\n"
        f"algo speedup={(t_py / t_cpp_bfs_cache):.2f}x"
    )

    assert speedup_union > 1.0 or speedup_bfs > 1.0