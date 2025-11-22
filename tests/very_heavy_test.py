# real-world simulations

import os
import time
import random
from typing import List, Tuple, Optional

import pytest
import networkx as nx

EARLY_STOP_SECONDS = 15 * 60  # 15 minutes

PR_NODES = 200  # nodes for PageRank induced subgraph
BC_NODES = 120  # nodes for betweenness induced subgraph
BC_K     = 8    # sampled k for betweenness
SEEDS    = 4    # bfs/dfs high-degree sources
OD_PAIRS = 8    # number of OD pairs (Dijkstra-only in hot paths)
ROUNDS   = 3    # scenario1 mutation rounds
MUT_ADD  = 20   # edges added per round
MUT_REM  = 20   # edges removed per round
MUT_TWK  = 60   # edges tweaked per round
ISO_N    = 40   # node count per iso slice
STEPS_S2 = 6    # scenario2 time steps (lighter than original)


def now() -> float:
    return time.perf_counter()

def elapsed(s: float) -> float:
    return time.perf_counter() - s

def maybe_stop(start_t: float) -> bool:
    return elapsed(start_t) >= EARLY_STOP_SECONDS

def undirected_copy_with_weights(G: nx.Graph) -> nx.Graph:
    UG = nx.Graph()
    UG.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        if UG.has_edge(u, v):
            if w < UG[u][v].get("weight", float("inf")):
                UG[u][v]["weight"] = w
        else:
            UG.add_edge(u, v, weight=w)
    return UG

def small_induced(G: nx.Graph, n: int, rng: random.Random) -> nx.Graph:
    nodes = list(G.nodes())
    if not nodes:
        return G.copy()
    rng.shuffle(nodes)
    n = max(1, min(n, len(nodes)))
    return G.subgraph(nodes[:n]).copy()

def random_nodes(G: nx.Graph, k: int, rng: random.Random) -> List:
    n = G.number_of_nodes()
    if n == 0:
        return []
    k = max(0, min(k, n))
    nodes = list(G.nodes())
    rng.shuffle(nodes)
    return nodes[:k]

def high_degree_nodes(G: nx.Graph, k: int) -> List:
    degs = sorted(G.degree, key=lambda x: x[1], reverse=True)
    return [n for n, _ in degs[:max(0, k)]]

def pick_od_pairs(G: nx.Graph, m: int, rng: random.Random) -> List[Tuple]:
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

def safe_shortest_path(
    G: nx.Graph,
    s,
    t,
    algorithm: str,
    backend: Optional[str] = None,
) -> Optional[List]:
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

def perform_mutations_weight_perturb(
    G: nx.Graph,
    rng: random.Random,
    add_edges: int = MUT_ADD,
    remove_edges: int = MUT_REM,
    tweak_edges: int = MUT_TWK,
):
    nodes = list(G.nodes())
    if not nodes:
        return

    # change a bunch of existing edges' weights slightly
    edges = list(G.edges())
    rng.shuffle(edges)
    for (u, v) in edges[:tweak_edges]:
        if G.has_edge(u, v):
            w = G[u][v].get("weight", 1.0)
            delta = w * (0.2 * (rng.random() - 0.5))
            G[u][v]["weight"] = max(1e-6, w + delta)

    # randomly remove some edges if available
    edges = list(G.edges())
    rng.shuffle(edges)
    for (u, v) in edges[:remove_edges]:
        if G.has_edge(u, v):
            G.remove_edge(u, v)

    # randomly add some edges with random weights
    high_deg = high_degree_nodes(G, min(200, max(10, len(nodes) // 10)))
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

def batch_pagerank(G: nx.Graph, backend: Optional[str], personalization=None):
    kw = dict(weight="weight")
    if backend:
        kw["backend"] = backend
    if personalization is not None:
        kw["personalization"] = personalization  # will trigger fallback in cpp path
    try:
        H = small_induced(G, PR_NODES, random)
        return nx.pagerank(H, **kw)
    except Exception:
        return None


def batch_bfs_dfs(G: nx.Graph, sources: List, backend: Optional[str]):
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
            # dfs with depth_limit to trigger fallback
            if backend:
                _ = list(nx.dfs_edges(G, s, backend=backend, depth_limit=3))
            else:
                _ = list(nx.dfs_edges(G, s, depth_limit=3))
        except Exception:
            pass

def batch_shortest_paths(
    G: nx.Graph,
    pairs: List[Tuple],
    backend: Optional[str],
    algorithms=("dijkstra",),
):
    _pairs = pairs[:OD_PAIRS]
    for (s, t) in _pairs:
        for algo in algorithms:
            _ = safe_shortest_path(G, s, t, algo, backend=backend)

def batch_mst_and_cc(G: nx.Graph, backend: Optional[str]):
    UG = undirected_copy_with_weights(G)
    try:
        if backend:
            _ = nx.minimum_spanning_tree(UG, weight="weight", backend=backend)
        else:
            _ = nx.minimum_spanning_tree(UG, weight="weight")
    except Exception:
        pass

    try:
        if backend:
            try:
                _ = list(nx.connected_components(UG, backend=backend))
            except Exception:
                _ = list(nx.connected_components(UG))
        else:
            _ = list(nx.connected_components(UG))
    except Exception:
        pass

def batch_betweenness(G: nx.Graph, rng: random.Random, backend: Optional[str]):
    H = small_induced(G, BC_NODES, rng)
    k = min(BC_K, max(1, H.number_of_nodes() // 4))
    try:
        if backend:
            _ = nx.betweenness_centrality(H, weight="weight", backend=backend, k=k, endpoints=True)
        else:
            _ = nx.betweenness_centrality(H, weight="weight", k=k, endpoints=True)
    except Exception:
        pass


def batch_isomorphic_checks(G1: nx.Graph, G2: nx.Graph, rng: random.Random, backend: Optional[str]):
    U1 = undirected_copy_with_weights(G1)
    U2 = undirected_copy_with_weights(G2)
    n1 = min(ISO_N, max(10, U1.number_of_nodes() // 100 or 10))
    n2 = min(ISO_N, max(10, U2.number_of_nodes() // 100 or 10))
    S1 = random_nodes(U1, n1, rng)
    S2 = random_nodes(U2, n2, rng)
    H1 = U1.subgraph(S1).copy()
    H2 = U2.subgraph(S2).copy()

    for i, n in enumerate(H1.nodes()):
        H1.nodes[n]["type"] = "a" if i % 2 == 0 else "b"
    for i, n in enumerate(H2.nodes()):
        H2.nodes[n]["type"] = "a" if i % 2 == 0 else "b"

    try:
        node_match = nx.algorithms.isomorphism.categorical_node_match("type", "a")
        if backend:
            _ = nx.is_isomorphic(H1, H2, node_match=node_match, backend=backend)
        else:
            _ = nx.is_isomorphic(H1, H2, node_match=node_match)
    except Exception:
        pass


def scenario1_workload(G_e, G_nyc, G_rome, rng: random.Random, backend: Optional[str], start_t: float):
    """Scenario 1: Multi-city maintenance & queries"""
    graphs = [("USA-E", G_e), ("NYC", G_nyc), ("ROME", G_rome)]

    # step 1: baseline metrics + traversals + paths
    for label, G in graphs:
        if maybe_stop(start_t): return
        _ = batch_pagerank(G, backend=backend)

        if maybe_stop(start_t): return
        seeds = high_degree_nodes(G, SEEDS)
        batch_bfs_dfs(G, seeds, backend=backend)

        if maybe_stop(start_t): return
        od_pairs = pick_od_pairs(G, OD_PAIRS, rng)
        batch_shortest_paths(G, od_pairs, backend=backend, algorithms=("dijkstra",))

        if maybe_stop(start_t): return
        batch_mst_and_cc(G, backend=backend)

        if maybe_stop(start_t): return
        batch_betweenness(G, rng, backend=backend)

        if maybe_stop(start_t): return
        try:
            seeds_small = random_nodes(G, 10, rng)
            personalization = {n: 1.0 for n in seeds_small}
            _ = batch_pagerank(G, backend=backend, personalization=personalization)
        except Exception:
            pass

    # step 2: mutations + recompute smaller workloads repeatedly
    for r in range(ROUNDS):
        if maybe_stop(start_t): return
        for label, G in graphs:
            if maybe_stop(start_t): return
            perform_mutations_weight_perturb(
                G, rng,
                add_edges=MUT_ADD,
                remove_edges=MUT_REM,
                tweak_edges=MUT_TWK
            )

            if maybe_stop(start_t): return
            seeds = high_degree_nodes(G, SEEDS)
            batch_bfs_dfs(G, seeds, backend=backend)

            if maybe_stop(start_t): return
            od_pairs = pick_od_pairs(G, OD_PAIRS, rng)
            batch_shortest_paths(G, od_pairs, backend=backend, algorithms=("dijkstra",))

            if maybe_stop(start_t): return
            batch_betweenness(G, rng, backend=backend)

            if maybe_stop(start_t): return
            other_label, other_G = random.choice(graphs)
            batch_isomorphic_checks(G, other_G, rng, backend=backend)


def scenario2_workload(G_e, G_nyc, G_rome, rng: random.Random, backend: Optional[str], start_t: float):
    """Scenario 2: Dynamic traffic simulation with rolling recomputations"""
    graphs = [("USA-E", G_e), ("NYC", G_nyc), ("ROME", G_rome)]

    steps = STEPS_S2
    for step in range(steps):
        if maybe_stop(start_t): return

        for label, G in graphs:
            if maybe_stop(start_t): return

            add = max(10, (80 if step < 3 else 40) // 2)
            rem = max(10, (80 if step < 3 else 40) // 2)
            tweak = max(60, (250 if step < 5 else 150) // 2)
            perform_mutations_weight_perturb(G, rng, add_edges=add, remove_edges=rem, tweak_edges=tweak)

            if maybe_stop(start_t): return
            m = min(OD_PAIRS * 2, 36 if step < 4 else 20)
            od_pairs = pick_od_pairs(G, m, rng)
            batch_shortest_paths(G, od_pairs, backend=backend, algorithms=("dijkstra",))

            if maybe_stop(start_t): return
            if step % 2 == 0:
                _ = batch_pagerank(G, backend=backend)
            else:
                seeds_small = random_nodes(G, 12, rng)
                personalization = {n: 1.0 for n in seeds_small}
                _ = batch_pagerank(G, backend=backend, personalization=personalization)  # may fallback

            if maybe_stop(start_t): return
            seeds = list({*(high_degree_nodes(G, 4)), *(random_nodes(G, 4, rng))})
            seeds = seeds[:SEEDS]
            batch_bfs_dfs(G, seeds, backend=backend)

            if maybe_stop(start_t): return
            if step % 3 == 0:
                batch_mst_and_cc(G, backend=backend)

            if maybe_stop(start_t): return
            batch_betweenness(G, rng, backend=backend)

            if maybe_stop(start_t): return
            if step % 2 == 1:
                other_label, other_G = random.choice(graphs)
                batch_isomorphic_checks(G, other_G, rng, backend=backend)


@pytest.mark.realworld
def test_realworld_scenario1_cpp(usa_e_graph, nyc_graph, rome_graph, rng_seed):
    G_nyc = nyc_graph
    G_rome = rome_graph
    G_e = usa_e_graph

    os.environ["OMP_NUM_THREADS"] = "4"
    random.seed(rng_seed)

    print("[Scenario 1 | CPP] Starting...")
    t0 = now()

    scenario1_workload(G_e, G_nyc, G_rome, random, backend="cpp", start_t=t0)

    total = elapsed(t0)
    if total >= EARLY_STOP_SECONDS:
        print("[Scenario 1 | CPP] Elapsed: 15+ minutes (stopped early)")
    else:
        print(f"[Scenario 1 | CPP] Elapsed: {total:.2f}s")


@pytest.mark.realworld
def test_realworld_scenario1_py(usa_e_graph, nyc_graph, rome_graph, rng_seed):
    G_nyc = nyc_graph
    G_rome = rome_graph
    G_e = usa_e_graph

    random.seed(rng_seed)

    print("[Scenario 1 | PY] Starting...")
    t0 = now()

    scenario1_workload(G_e, G_nyc, G_rome, random, backend=None, start_t=t0)

    total = elapsed(t0)
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
    t0 = now()

    scenario2_workload(G_e, G_nyc, G_rome, random, backend="cpp", start_t=t0)

    total = elapsed(t0)
    if total >= EARLY_STOP_SECONDS:
        print("[Scenario 2 | CPP] Elapsed: 15+ minutes (stopped early)")
    else:
        print(f"[Scenario 2 | CPP] Elapsed: {total:.2f}s")


@pytest.mark.realworld
def test_realworld_scenario2_py(usa_e_graph, nyc_graph, rome_graph, rng_seed):
    G_e = usa_e_graph
    G_nyc = nyc_graph
    G_rome = rome_graph

    random.seed(rng_seed)

    print("[Scenario 2 | PY] Starting...")
    t0 = now()

    scenario2_workload(G_e, G_nyc, G_rome, random, backend=None, start_t=t0)

    total = elapsed(t0)
    if total >= EARLY_STOP_SECONDS:
        print("[Scenario 2 | PY] Elapsed: 15+ minutes (stopped early)")
    else:
        print(f"[Scenario 2 | PY] Elapsed: {total:.2f}s")