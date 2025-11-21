import time
import networkx as nx
import pytest

# still developing --> meant to be a test that incoroporates multiple cpp calls to all functions to see caching benefits
@pytest.mark.usa_e
@pytest.mark.performance
@pytest.mark.slow
def test_heavy(usa_e_graph):
    G = usa_e_graph
    source = next(iter(G.nodes))

    t0 = time.perf_counter()
    _ = nx.pagerank(G, backend="cpp")
    t_cpp = time.perf_counter() - t0

    print("")
    print(t_cpp)

    t0 = time.perf_counter()
    _ = nx.pagerank(G, backend="cpp")
    t_cpp_cache = time.perf_counter() - t0
    print(t_cpp_cache)

    t0 = time.perf_counter()
    _ = list(nx.dfs_edges(G, source, backend="cpp"))
    t_cpp = time.perf_counter() - t0
    print(t_cpp)

    t0 = time.perf_counter()
    _ = list(nx.dfs_edges(G, source, backend="cpp"))
    t_cpp_cache = time.perf_counter() - t0
    print(t_cpp_cache)

    t0 = time.perf_counter()
    _ = list(nx.bfs_edges(G, source, backend="cpp"))
    t_cpp = time.perf_counter() - t0
    print(t_cpp)

    t0 = time.perf_counter()
    _ = list(nx.bfs_edges(G, source, backend="cpp"))
    t_cpp_cache = time.perf_counter() - t0
    print(t_cpp_cache)

    t0 = time.perf_counter()
    _ = nx.pagerank(G)
    t_cpp = time.perf_counter() - t0

    print("")
    print(t_cpp)

    t0 = time.perf_counter()
    _ = nx.pagerank(G)
    t_cpp_cache = time.perf_counter() - t0
    print(t_cpp_cache)

    t0 = time.perf_counter()
    _ = list(nx.dfs_edges(G, source))
    t_cpp = time.perf_counter() - t0
    print(t_cpp)

    t0 = time.perf_counter()
    _ = list(nx.dfs_edges(G, source))
    t_cpp_cache = time.perf_counter() - t0
    print(t_cpp_cache)

    t0 = time.perf_counter()
    _ = list(nx.bfs_edges(G, source))
    t_cpp = time.perf_counter() - t0
    print(t_cpp)

    t0 = time.perf_counter()
    _ = list(nx.bfs_edges(G, source))
    t_cpp_cache = time.perf_counter() - t0
    print(t_cpp_cache)