# very_heavy_test.py
# Real-world simulations (timing-only) in a *light* configuration.
# Four tests total (two scenarios × cpp & python), each marked @pytest.mark.realworld.
# Timer starts AFTER fixtures are assigned. Early-stop at 15 minutes per test.
#
# This file is the "light mode" permanently (no flags/toggles). It:
#   • Uses smaller induced subgraphs for expensive ops (PageRank, betweenness)
#   • Uses fewer OD pairs (Dijkstra-only in the hot paths)
#   • Uses fewer BFS/DFS seeds and lighter mutations
#   • Uses smaller isomorphism slices
#   • Keeps realistic flows and exercises the C++-backed APIs
#
# Requirements from your setup:
#   • All three fixtures are directed & weighted graphs (no multigraphs)
#   • For weighted ops, we always pass weight="weight"
#   • For shortest_path we pass method="dijkstra" or "bellman-ford"
#   • For C++ runs we pass backend="cpp" only on supported calls
#   • For is_isomorphic in C++ runs, pass backend="cpp" (per your request)

import os
import time
import random
from typing import List, Tuple, Optional

import pytest
import networkx as nx


# --------------------------
# Global timing (fixed)
# --------------------------

EARLY_STOP_SECONDS = 15 * 60  # 15 minutes

# Light behavior constants (inlined; no flags)
PR_NODES = 200          # nodes for PageRank induced subgraph
BC_NODES = 120          # nodes for betweenness induced subgraph
BC_K      = 8           # sampled k for betweenness
SEEDS     = 4           # bfs/dfs high-degree sources
OD_PAIRS  = 8           # number of OD pairs (Dijkstra-only in hot paths)
ROUNDS    = 3           # scenario1 mutation rounds
MUT_ADD   = 20          # edges added per round
MUT_REM   = 20          # edges removed per round
MUT_TWK   = 60          # edges tweaked per round
ISO_N     = 40          # node count per iso slice
STEPS_S2  = 6           # scenario2 time steps (lighter than original)


# --------------------------
# Utilities (no assertions)
# --------------------------

def _now() -> float:
    return time.perf_counter()


def _elapsed(s: float) -> float:
    return time.perf_counter() - s


def _maybe_stop(start_t: float) -> bool:
    """Return True if we've exceeded the early-stop budget."""
    return _elapsed(start_t) >= EARLY_STOP_SECONDS


def _undirected_copy_with_weights(G: nx.Graph) -> nx.Graph:
    """Make an undirected simple graph with the same 'weight' attributes (if present)."""
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes(data=True))
    # If both u->v and v->u exist, keep min weight for MST sanity.
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        if UG.has_edge(u, v):
            if w < UG[u][v].get("weight", float("inf")):
                UG[u][v]["weight"] = w
        else:
            UG.add_edge(u, v, weight=w)
    return UG


def _small_induced(G: nx.Graph, n: int, rng: random.Random) -> nx.Graph:
    """Induced subgraph of ~n nodes (bounded by graph size)."""
    nodes = list(G.nodes())
    if not nodes:
        return G.copy()
    rng.shuffle(nodes)
    n = max(1, min(n, len(nodes)))
    return G.subgraph(nodes[:n]).copy()


def _random_nodes(G: nx.Graph, k: int, rng: random.Random) -> List:
    n = G.number_of_nodes()
    if n == 0:
        return []
    k = max(0, min(k, n))
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    return nodes[:k]


def _high_degree_nodes(G: nx.Graph, k: int) -> List:
    degs = sorted(G.degree, key=lambda x: x[1], reverse=True)
    return [n for n, _ in degs[:max(0, k)]]


def _pick_od_pairs(G: nx.Graph, m: int, rng: random.Random) -> List[Tuple]:
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return []
    rng.shuffle(nodes)
    pairs = []
    i = 0
    while len(pairs) < m and i + 1 < len(nodes):
        s = nodes[i]
        t = nodes[i + 1]
        if s != t:
            pairs.append((s, t))
        i += 2
    return pairs


def _safe_shortest_path(
    G: nx.Graph,
    s,
    t,
    algorithm: str,
    backend: Optional[str] = None,
) -> Optional[List]:
    """Shortest path wrapper using NetworkX's 'method=' kw (dijkstra / bellman-ford)."""
    try:
        if algorithm == "dijkstra":
            if backend:
                return nx.shortest_path(G, source=s, target=t, weight="weight", method="dijkstra", backend=backend)
            else:
                return nx.shortest_path(G, source=s, target=t, weight="weight", method="dijkstra")
        elif algorithm == "bellman-ford":
            if backend:
                return nx.shortest_path(G, source=s, target=t, weight="weight", method="bellman-ford", backend=backend)
            else:
                return nx.shortest_path(G, source=s, target=t, weight="weight", method="bellman-ford")
    except Exception:
        return None
    return None


def _perform_mutations_weight_perturb(
    G: nx.Graph,
    rng: random.Random,
    add_edges: int = MUT_ADD,
    remove_edges: int = MUT_REM,
    tweak_edges: int = MUT_TWK,
):
    """Random small mutations: add, remove, and tweak edge weights."""
    nodes = list(G.nodes())
    if not nodes:
        return

    # Tweak a bunch of existing edges' weights slightly
    edges = list(G.edges())
    rng.shuffle(edges)
    for (u, v) in edges[:tweak_edges]:
        if G.has_edge(u, v):
            w = G[u][v].get("weight", 1.0)
            # Perturb within +/-10%
            delta = w * (0.2 * (rng.random() - 0.5))
            G[u][v]["weight"] = max(1e-6, w + delta)

    # Randomly remove some edges if available
    edges = list(G.edges())
    rng.shuffle(edges)
    for (u, v) in edges[:remove_edges]:
        if G.has_edge(u, v):
            G.remove_edge(u, v)

    # Randomly add some edges with random weights (prefer connecting high-degree nodes)
    high_deg = _high_degree_nodes(G, min(200, max(10, len(nodes) // 10)))
    pool = high_deg if len(high_deg) >= 2 else nodes
    for _ in range(add_edges):
        if len(pool) < 2:
            break
        u = rng.choice(pool)
        v = rng.choice(pool)
        if u == v:
            continue
        if not G.has_edge(u, v):
            G.add_edge(u, v, weight=max(1e-3, 1.0 + 10.0 * rng.random()))


def _batch_pagerank(G: nx.Graph, backend: Optional[str], personalization=None):
    kw = dict(weight="weight")
    if backend:
        kw["backend"] = backend
    if personalization is not None:
        kw["personalization"] = personalization  # may trigger fallback in cpp path
    try:
        # Always run PageRank on a small induced slice to reduce runtime.
        H = _small_induced(G, PR_NODES, random)
        return nx.pagerank(H, **kw)
    except Exception:
        return None


def _batch_bfs_dfs(G: nx.Graph, sources: List, backend: Optional[str]):
    # Trim source list to SEEDS
    srcs = sources[:SEEDS]
    for s in srcs:
        try:
            # bfs
            if backend:
                _ = list(nx.bfs_edges(G, s, backend=backend))
            else:
                _ = list(nx.bfs_edges(G, s))
        except Exception:
            pass
        try:
            # dfs (with depth_limit to possibly trigger fallback when in cpp)
            if backend:
                _ = list(nx.dfs_edges(G, s, backend=backend, depth_limit=3))
            else:
                _ = list(nx.dfs_edges(G, s, depth_limit=3))
        except Exception:
            pass


def _batch_shortest_paths(
    G: nx.Graph,
    pairs: List[Tuple],
    backend: Optional[str],
    algorithms=("dijkstra",),
):
    # Fewer OD pairs; Dijkstra-only here
    _pairs = pairs[:OD_PAIRS]
    for (s, t) in _pairs:
        for algo in algorithms:
            _ = _safe_shortest_path(G, s, t, algo, backend=backend)


def _batch_mst_and_cc(G: nx.Graph, backend: Optional[str]):
    UG = _undirected_copy_with_weights(G)
    try:
        if backend:
            _ = nx.minimum_spanning_tree(UG, weight="weight", backend=backend)
        else:
            _ = nx.minimum_spanning_tree(UG, weight="weight")
    except Exception:
        pass

    try:
        # connected_components is for undirected graphs in NetworkX
        if backend:
            # Some backends may not implement CC; call without backend if it errors.
            try:
                _ = list(nx.connected_components(UG, backend=backend))  # if supported
            except Exception:
                _ = list(nx.connected_components(UG))
        else:
            _ = list(nx.connected_components(UG))
    except Exception:
        pass


def _batch_betweenness(G: nx.Graph, rng: random.Random, backend: Optional[str]):
    # Use a small induced subgraph & small k
    H = _small_induced(G, BC_NODES, rng)
    k = min(BC_K, max(1, H.number_of_nodes() // 4))
    # Keep endpoints=True to preserve fallback flavor if cpp lacks that param
    try:
        if backend:
            _ = nx.betweenness_centrality(H, weight="weight", backend=backend, k=k, endpoints=True)
        else:
            _ = nx.betweenness_centrality(H, weight="weight", k=k, endpoints=True)
    except Exception:
        pass


def _batch_isomorphic_checks(G1: nx.Graph, G2: nx.Graph, rng: random.Random, backend: Optional[str]):
    # Compare small undirected induced subgraphs; include node_match to trigger fallback.
    U1 = _undirected_copy_with_weights(G1)
    U2 = _undirected_copy_with_weights(G2)
    n1 = min(ISO_N, max(10, U1.number_of_nodes() // 100 or 10))
    n2 = min(ISO_N, max(10, U2.number_of_nodes() // 100 or 10))
    S1 = _random_nodes(U1, n1, rng)
    S2 = _random_nodes(U2, n2, rng)
    H1 = U1.subgraph(S1).copy()
    H2 = U2.subgraph(S2).copy()

    # Inject simple attributes for node_match
    for i, n in enumerate(H1.nodes()):
        H1.nodes[n]["type"] = "a" if i % 2 == 0 else "b"
    for i, n in enumerate(H2.nodes()):
        H2.nodes[n]["type"] = "a" if i % 2 == 0 else "b"

    try:
        node_match = nx.algorithms.isomorphism.categorical_node_match("type", "a")
        # Pass backend="cpp" only in cpp runs per your request
        if backend:
            _ = nx.is_isomorphic(H1, H2, node_match=node_match, backend=backend)
        else:
            _ = nx.is_isomorphic(H1, H2, node_match=node_match)
    except Exception:
        pass


# --------------------------
# Scenario workloads
# --------------------------

def _scenario1_workload(G_e, G_nyc, G_rome, rng: random.Random, backend: Optional[str], start_t: float):
    """Scenario 1: Multi-city maintenance & queries (light)."""
    graphs = [("USA-E", G_e), ("NYC", G_nyc), ("ROME", G_rome)]

    # Pass 1: Baseline metrics + traversals + paths
    for label, G in graphs:
        if _maybe_stop(start_t): return
        # Pagerank baseline on a small induced slice
        _ = _batch_pagerank(G, backend=backend)

        if _maybe_stop(start_t): return
        # BFS/DFS from high-degree seeds
        seeds = _high_degree_nodes(G, SEEDS)
        _batch_bfs_dfs(G, seeds, backend=backend)

        if _maybe_stop(start_t): return
        # Shortest paths on random OD pairs (Dijkstra only here)
        od_pairs = _pick_od_pairs(G, OD_PAIRS, rng)
        _batch_shortest_paths(G, od_pairs, backend=backend, algorithms=("dijkstra",))

        if _maybe_stop(start_t): return
        # MST + connected components on undirected copy
        _batch_mst_and_cc(G, backend=backend)

        if _maybe_stop(start_t): return
        # Betweenness on small induced subgraph
        _batch_betweenness(G, rng, backend=backend)

        if _maybe_stop(start_t): return
        # Graceful fallback sampler: pagerank with personalization (small set)
        try:
            seeds_small = _random_nodes(G, 10, rng)
            personalization = {n: 1.0 for n in seeds_small}
            _ = _batch_pagerank(G, backend=backend, personalization=personalization)
        except Exception:
            pass

    # Pass 2: Mutations + recompute smaller workloads repeatedly (light)
    for r in range(ROUNDS):
        if _maybe_stop(start_t): return
        for label, G in graphs:
            if _maybe_stop(start_t): return
            _perform_mutations_weight_perturb(
                G, rng,
                add_edges=MUT_ADD,
                remove_edges=MUT_REM,
                tweak_edges=MUT_TWK
            )

            if _maybe_stop(start_t): return
            # Quick recompute batches to reflect new topology/weights
            seeds = _high_degree_nodes(G, SEEDS)
            _batch_bfs_dfs(G, seeds, backend=backend)

            if _maybe_stop(start_t): return
            od_pairs = _pick_od_pairs(G, OD_PAIRS, rng)
            _batch_shortest_paths(G, od_pairs, backend=backend, algorithms=("dijkstra",))

            if _maybe_stop(start_t): return
            _batch_betweenness(G, rng, backend=backend)

            if _maybe_stop(start_t): return
            # Occasional isomorphism checks on slices across graphs to exercise API
            # Pick another graph for cross-compare
            other_label, other_G = random.choice(graphs)
            _batch_isomorphic_checks(G, other_G, rng, backend=backend)


def _scenario2_workload(G_e, G_nyc, G_rome, rng: random.Random, backend: Optional[str], start_t: float):
    """Scenario 2: Dynamic traffic simulation with rolling recomputations (light)."""
    graphs = [("USA-E", G_e), ("NYC", G_nyc), ("ROME", G_rome)]

    steps = STEPS_S2
    for step in range(steps):
        if _maybe_stop(start_t): return

        for label, G in graphs:
            if _maybe_stop(start_t): return

            # Smaller perturbations than original
            add = max(10, (80 if step < 3 else 40) // 2)
            rem = max(10, (80 if step < 3 else 40) // 2)
            tweak = max(60, (250 if step < 5 else 150) // 2)
            _perform_mutations_weight_perturb(G, rng, add_edges=add, remove_edges=rem, tweak_edges=tweak)

            if _maybe_stop(start_t): return
            # Rotating OD pairs — modest count
            m = min(OD_PAIRS * 2, 36 if step < 4 else 20)
            od_pairs = _pick_od_pairs(G, m, rng)
            # Keep Dijkstra here; optional Bellman-Ford could be added on a tiny slice if desired
            _batch_shortest_paths(G, od_pairs, backend=backend, algorithms=("dijkstra",))

            if _maybe_stop(start_t): return
            # Pagerank every step; every other step, do a fallback-flavored one
            if step % 2 == 0:
                _ = _batch_pagerank(G, backend=backend)
            else:
                seeds_small = _random_nodes(G, 12, rng)
                personalization = {n: 1.0 for n in seeds_small}
                _ = _batch_pagerank(G, backend=backend, personalization=personalization)  # may fallback

            if _maybe_stop(start_t): return
            # Frequent BFS/DFS, seed from mixed high-degree/random
            seeds = list({*(_high_degree_nodes(G, 4)), *(_random_nodes(G, 4, rng))})
            seeds = seeds[:SEEDS]
            _batch_bfs_dfs(G, seeds, backend=backend)

            if _maybe_stop(start_t): return
            # Every 3 steps: CC + MST on undirected snapshot
            if step % 3 == 0:
                _batch_mst_and_cc(G, backend=backend)

            if _maybe_stop(start_t): return
            # Every step: betweenness on rotating small slice
            _batch_betweenness(G, rng, backend=backend)

            if _maybe_stop(start_t): return
            # Periodic isomorphism checks (with node_match) against another city snapshot
            if step % 2 == 1:
                other_label, other_G = random.choice(graphs)
                _batch_isomorphic_checks(G, other_G, rng, backend=backend)


# --------------------------
# Tests (duplicate bodies)
# --------------------------

@pytest.mark.realworld
def test_realworld_scenario1_cpp(usa_e_graph, nyc_graph, rome_graph, rng_seed):
    # Graphs are already loaded by fixtures; assign first, then start timer.
    G_e = usa_e_graph
    G_nyc = nyc_graph
    G_rome = rome_graph

    # Environment
    os.environ["OMP_NUM_THREADS"] = "4"
    random.seed(rng_seed)

    print("[Scenario 1 | CPP] Starting...")
    t0 = _now()

    # Workload
    _scenario1_workload(G_e, G_nyc, G_rome, random, backend="cpp", start_t=t0)

    total = _elapsed(t0)
    if total >= EARLY_STOP_SECONDS:
        print("[Scenario 1 | CPP] Elapsed: 15+ minutes (stopped early)")
    else:
        print(f"[Scenario 1 | CPP] Elapsed: {total:.2f}s")


@pytest.mark.realworld
def test_realworld_scenario1_py(usa_e_graph, nyc_graph, rome_graph, rng_seed):
    G_e = usa_e_graph
    G_nyc = nyc_graph
    G_rome = rome_graph

    os.environ["OMP_NUM_THREADS"] = "4"
    random.seed(rng_seed)

    print("[Scenario 1 | PY] Starting...")
    t0 = _now()

    _scenario1_workload(G_e, G_nyc, G_rome, random, backend=None, start_t=t0)

    total = _elapsed(t0)
    if total >= EARLY_STOP_SECONDS:
        print("[Scenario 1 | PY] Elapsed: 15+ minutes (stopped early)")
    else:
        print(f"[Scenario 1 | PY] Elapsed: {total:.2f}s")


@pytest.mark.realworld
def test_realworld_scenario2_cpp(usa_e_graph, nyc_graph, rome_graph, rng_seed):
    G_e = usa_e_graph
    G_nyc = nyc_graph
    G_rome = rome_graph

    os.environ["OMP_NUM_THREADS"] = "4"
    random.seed(rng_seed)

    print("[Scenario 2 | CPP] Starting...")
    t0 = _now()

    _scenario2_workload(G_e, G_nyc, G_rome, random, backend="cpp", start_t=t0)

    total = _elapsed(t0)
    if total >= EARLY_STOP_SECONDS:
        print("[Scenario 2 | CPP] Elapsed: 15+ minutes (stopped early)")
    else:
        print(f"[Scenario 2 | CPP] Elapsed: {total:.2f}s")


@pytest.mark.realworld
def test_realworld_scenario2_py(usa_e_graph, nyc_graph, rome_graph, rng_seed):
    G_e = usa_e_graph
    G_nyc = nyc_graph
    G_rome = rome_graph

    os.environ["OMP_NUM_THREADS"] = "4"
    random.seed(rng_seed)

    print("[Scenario 2 | PY] Starting...")
    t0 = _now()

    _scenario2_workload(G_e, G_nyc, G_rome, random, backend=None, start_t=t0)

    total = _elapsed(t0)
    if total >= EARLY_STOP_SECONDS:
        print("[Scenario 2 | PY] Elapsed: 15+ minutes (stopped early)")
    else:
        print(f"[Scenario 2 | PY] Elapsed: {total:.2f}s")