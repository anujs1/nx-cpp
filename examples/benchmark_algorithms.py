#!/usr/bin/env python3
"""
Deterministic micro-benchmarks for core algorithms implemented in nx_cpp._nx_cpp.

- Builds a fixed random graph with a provided seed
- Converts it once to the C++ graph to exclude conversion cost
- Times each algorithm over repeated runs with warmups

Algorithms timed:
- PageRank
- BFS (parent array)
- DFS (parent array)
- Dijkstra (distances + parents)
- Bellman-Ford (distances + parents)
- Betweenness centrality (unweighted Brandes)

Example:
  python3 examples/benchmark_algorithms.py --n 2000 --m 8000 --repeats 5 --warmup 2
"""

import argparse
import statistics
import time
from typing import Dict, Callable

import networkx as nx

from nx_cpp.backend import convert_from_nx
from nx_cpp._nx_cpp import (
    pagerank as c_pagerank,
    bfs_edges as c_bfs_edges,
    dfs_edges as c_dfs_edges,
    dijkstra as c_dijkstra,
    bellman_ford as c_bellman_ford,
    betweenness_centrality as c_betweenness,
)


def deterministic_weight(u: int, v: int) -> float:
    """Deterministic pseudo-weight in [1.0, 10.0]."""
    x = (u * 1315423911) ^ (v * 2654435761)
    val = (x & 0xFF) / 255.0
    return 1.0 + 9.0 * val


def make_graph(n: int, m: int, directed: bool, seed: int, weighted: bool) -> nx.Graph:
    G = nx.gnm_random_graph(n, m, seed=seed, directed=directed)
    if weighted:
        for u, v in G.edges():
            G.edges[u, v]["weight"] = float(deterministic_weight(int(u), int(v)))
    return G


def time_fn(fn: Callable[[], None], repeats: int, warmup: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    return {
        "runs": repeats,
        "min": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "stdev": statistics.pstdev(samples) if repeats > 1 else 0.0,
    }


def fmt_ms(sec: float) -> str:
    return f"{sec * 1000.0:.3f} ms"


def main() -> None:
    ap = argparse.ArgumentParser(description="nx-cpp deterministic algorithm microbenchmarks")
    ap.add_argument("--n", type=int, default=1000, help="number of nodes")
    ap.add_argument("--m", type=int, default=4000, help="number of edges")
    ap.add_argument("--directed", action="store_true", help="use directed graph (default: undirected)")
    ap.add_argument("--seed", type=int, default=42, help="random seed for graph generation")
    ap.add_argument("--weighted", action="store_true", help="add deterministic weights to edges")
    ap.add_argument("--repeats", type=int, default=5, help="timed runs per algorithm")
    ap.add_argument("--warmup", type=int, default=2, help="warmup runs per algorithm (not timed)")
    ap.add_argument("--alpha", type=float, default=0.85, help="PageRank damping factor")
    ap.add_argument("--max_iter", type=int, default=50, help="PageRank max iterations")
    ap.add_argument("--tol", type=float, default=1e-6, help="PageRank tolerance")
    ap.add_argument("--source", type=int, default=0, help="source node index for BFS/DFS/SP")

    args = ap.parse_args()

    print("=== Graph Setup ===")
    print(f"n={args.n}, m={args.m}, directed={args.directed}, weighted={args.weighted}, seed={args.seed}")

    # Build NetworkX graph deterministically
    G = make_graph(args.n, args.m, directed=args.directed, seed=args.seed, weighted=args.weighted)

    # Convert once; conversion time is printed by the backend but excluded from timings
    nxcpp = convert_from_nx(G, weight="weight", use_cache=False)
    CG = nxcpp._G  # underlying C++ Graph

    src = int(args.source)
    if src < 0 or src >= args.n:
        raise SystemExit(f"Invalid --source {src}; must be in [0, {args.n-1}]")

    print("\n=== Benchmarking (conversion excluded) ===")

    # Define runnable closures per algorithm
    runners: Dict[str, Callable[[], None]] = {
        "pagerank": lambda: c_pagerank(graph=CG, alpha=args.alpha, max_iter=args.max_iter, tol=args.tol),
        "bfs": lambda: c_bfs_edges(graph=CG, source=src),
        "dfs": lambda: c_dfs_edges(graph=CG, source=src),
        "dijkstra": lambda: c_dijkstra(graph=CG, source=src),
        "bellman_ford": lambda: c_bellman_ford(graph=CG, source=src),
        "betweenness": lambda: c_betweenness(graph=CG, normalized=True, endpoints=False),
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, fn in runners.items():
        res = time_fn(fn, repeats=args.repeats, warmup=args.warmup)
        results[name] = res
        print(f"- {name:14s} median {fmt_ms(res['median'])} \t(min {fmt_ms(res['min'])}, runs={res['runs']})")

    print("\nDone. Use identical args across runs to compare optimizations deterministically.")


if __name__ == "__main__":
    main()
