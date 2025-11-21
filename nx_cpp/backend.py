import sys
import time
from collections import defaultdict

import networkx as nx

# pybind11 module
from ._nx_cpp import (
    Graph as CppGraph,
    pagerank as _cpp_pagerank,
    bfs_edges as _cpp_bfs_edges,
    dfs_edges as _cpp_dfs_edges,
    dijkstra as _cpp_dijkstra,
    bellman_ford as _cpp_bellman_ford,
    unweighted_shortest_path as _cpp_unweighted_shortest_path,
    betweenness_centrality as _cpp_betweenness_centrality,
    connected_components_union_find as _cpp_connected_components_union_find,
    connected_components_bfs as _cpp_connected_components_bfs,
    minimum_spanning_tree as _cpp_minimum_spanning_tree,
    graphs_are_isomorphic as _cpp_graphs_are_isomorphic,
)
class NxCppGraph:
    """
    Minimal backend graph wrapper understood by NetworkX dispatch
    Holds a C++ Graph and preserves original Python node labels
    """

    __networkx_backend__ = "cpp"

    def __init__(self, cpp_graph: CppGraph, nodes, orig_graph=None):
        self._G = cpp_graph
        self._nodes = list(nodes)
        self._orig_graph = orig_graph
        # Fast label->index map for non-integer / string labels
        try:
            self._index = {n: i for i, n in enumerate(self._nodes)}
        except Exception:
            self._index = {}

    def is_directed(self) -> bool:
        return self._G.is_directed()

    def is_multigraph(self) -> bool:
        return self._G.is_multigraph()

    def _edges_py(self):
        idx_edges = self._G.edges()
        nodes = self._nodes
        return [(nodes[u], nodes[v]) for (u, v) in idx_edges]


def _component_sets_from_ids(nodes, component_ids):
    groups = defaultdict(set)
    for idx, comp_id in enumerate(component_ids):
        groups[int(comp_id)].add(nodes[idx])
    sorted_components = sorted(groups.values(), key=len, reverse=True)
    for component in sorted_components:
        yield component


def convert_from_nx(G, weight='weight', **kwargs):
    """
    Convert a NetworkX Graph -> NxCppGraph
    - Nodes are relabeled to 0..n-1 for C++
    - Parallel edges collapsed (no multigraph support)
    - Supports weighted graphs via weight parameter
    """
    # t_start = time.time()
    nodes = list(G.nodes())
    # t_nodes = time.time()
    
    index = {n: i for i, n in enumerate(nodes)}
    directed = G.is_directed()
    
    # Check if graph has weights
    has_weights = False
    if G.number_of_edges() > 0:
        # Sample first edge to check for weights
        for u, v, data in G.edges(data=True):
            if weight in data:
                has_weights = True
            break

    if has_weights:
        # Create weighted edge list
        if G.is_multigraph():
            edge_dict = {}
            for u, v, k, data in G.edges(keys=True, data=True):
                edge = (index[u], index[v])
                w = data.get(weight, 1.0)
                if edge not in edge_dict or w < edge_dict[edge]:
                    edge_dict[edge] = w
            edges = [(u, v, w) for (u, v), w in edge_dict.items()]
        else:
            edges = [(index[u], index[v], data.get(weight, 1.0)) 
                    for u, v, data in G.edges(data=True)]
    else:
        # Create unweighted edge list
        if G.is_multigraph():
            edges = {(index[u], index[v]) for (u, v, _k) in G.edges(keys=True)}
            edges = list(edges)
        else:
            edges = [(index[u], index[v]) for (u, v) in G.edges()]
    
    # t_edges = time.time()
    cpp_graph = CppGraph(len(nodes), edges, directed)
    # t_cpp = time.time()
    
    # print("")
    # print(f"[Conversion] nodes: {t_nodes - t_start:.3f}s, edges: {t_edges - t_nodes:.3f}s, cpp_graph: {t_cpp - t_edges:.3f}s, total: {t_cpp - t_start:.3f}s")
    
    return NxCppGraph(cpp_graph, nodes, orig_graph=G)


def convert_to_nx(obj, **kwargs):
    """NxCppGraph -> NetworkX Graph/DiGraph (no attributes)"""
    if isinstance(obj, NxCppGraph):
        if getattr(obj, "_orig_graph", None) is not None:
            return obj._orig_graph
        H = nx.DiGraph() if obj.is_directed() else nx.Graph()
        H.add_nodes_from(obj._nodes)
        H.add_edges_from(obj._edges_py())
        return H
    if isinstance(obj, CppGraph):
        H = nx.DiGraph() if obj.is_directed() else nx.Graph()
        # can't recover nodes here, but should always be getting an NxCppGraph anyway
        H.add_edges_from(obj.edges())
        return H
    return obj


def can_run(name, args, kwargs):
    return name in (
        "pagerank",
        "bfs_edges",
        "dfs_edges",
        "shortest_path",
        "betweenness_centrality",
        "connected_components",
        "minimum_spanning_tree",
        "is_isomorphic",
    )


def should_run(name, args, kwargs):
    # defaulting to True for all supported functions for testing purposes
    return True
    # """Prefer C++ when the graph is not tiny"""
    # G = args[0] if args else kwargs.get("G", None)
    # try:
    #     n = len(G._nodes) if isinstance(G, NxCppGraph) else G.number_of_nodes()
    # except Exception:
    #     n = 0
    # return n >= 50


def pagerank(
    G,
    alpha: float = 0.85,
    personalization=None,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    nstart=None,
    weight='weight',
    dangling=None,
    *args,
    **kwargs,
):
    """
    Backend implementation for nx.pagerank
    Only alpha, max_iter, weight, tol are used; other kwargs being used leads to Python fallback
    """
    if personalization is not None or nstart is not None or (weight != 'weight' and weight is not None) or dangling is not None:
        G = convert_to_nx(G)
        return nx.pagerank(
            G,
            alpha=alpha,
            personalization=personalization,
            max_iter=max_iter,
            tol=tol,
            nstart=nstart,
            weight=weight,
            dangling=dangling
        )
    try:
        if isinstance(G, NxCppGraph):
            pr = _cpp_pagerank(graph=G._G, alpha=alpha, max_iter=max_iter, tol=tol)
            return {node: float(pr[i]) for i, node in enumerate(G._nodes)}
        elif isinstance(G, CppGraph):
            pr = _cpp_pagerank(graph=G, alpha=alpha, max_iter=max_iter, tol=tol)
            return {i: float(pr[i]) for i in range(len(pr))}
        else:
            # return pagerank(convert_from_nx(G), alpha=alpha, max_iter=max_iter, tol=tol)
            # if dispatcher called this function, we will always get an NxCppGraph
            # if we reach here, that means the user directly called the nx_cpp.backend.function with a NetworkX graph, so we use the Python backend
            return nx.pagerank(
                G,
                alpha=alpha,
                personalization=personalization,
                max_iter=max_iter,
                tol=tol,
                nstart=nstart,
                weight=weight,
                dangling=dangling
            )
    except Exception:
        G = convert_to_nx(G)
        return nx.pagerank(
            G,
            alpha=alpha,
            personalization=personalization,
            max_iter=max_iter,
            tol=tol,
            nstart=nstart,
            weight=weight,
            dangling=dangling
        )


def bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    """
    Backend implementation for nx.bfs_edges
    Only basic BFS is supported; other kwargs being used leads to Python fallback
    Returns an iterator of edges in BFS order from source
    """
    if depth_limit is not None or sort_neighbors is not None or reverse:
        G = convert_to_nx(G)
        return nx.bfs_edges(
            G,
            source=source,
            reverse=reverse,
            depth_limit=depth_limit,
            sort_neighbors=sort_neighbors
        )
    try:
        if isinstance(G, NxCppGraph):
            nodes = G._nodes
            source_idx = G._index.get(source, -1)
            if source_idx == -1:
                raise nx.NodeNotFound(f"Source node {source} not in graph")
            
            raw_edges = _cpp_bfs_edges(graph=G._G, source=source_idx)
            edges = []
            for u_idx, v_idx in raw_edges:
                if u_idx != -1:
                    u, v = nodes[u_idx], nodes[v_idx]
                    edges.append((u, v) if not reverse else (v, u))
            return iter(edges)
        elif isinstance(G, CppGraph):
            # if source < 0 or source >= len(G.edges()):
            if source < 0 or source >= G.n:
                raise nx.NodeNotFound(f"Source node {source} not in graph")
            
            raw_edges = _cpp_bfs_edges(graph=G, source=source)
            edges = []
            for u, v in raw_edges:
                if u != -1:
                    edges.append((u, v) if not reverse else (v, u))
            return iter(edges)
        else:
            # return bfs_edges(convert_from_nx(G), source=source, reverse=reverse, 
                            # depth_limit=depth_limit, sort_neighbors=sort_neighbors)
            # if dispatcher called this function, we will always get an NxCppGraph
            # if we reach here, that means the user directly called the nx_cpp.backend.function with a NetworkX graph, so we use the Python backend
            return nx.bfs_edges(
                G,
                source=source,
                reverse=reverse,
                depth_limit=depth_limit,
                sort_neighbors=sort_neighbors
            )
            
    except Exception:
        G = convert_to_nx(G)
        return nx.bfs_edges(
            G,
            source=source,
            reverse=reverse,
            depth_limit=depth_limit,
            sort_neighbors=sort_neighbors
        )


def dfs_edges(G, source, depth_limit=None, sort_neighbors=None):
    """
    Backend implementation for nx.dfs_edges
    Only basic DFS is supported; other kwargs being used leads to Python fallback
    Returns an iterator of edges in DFS order from source
    """
    if depth_limit is not None or sort_neighbors is not None:
        G = convert_to_nx(G)
        return nx.dfs_edges(G, source=source, depth_limit=depth_limit, sort_neighbors=sort_neighbors)
    try:
        if isinstance(G, NxCppGraph):
            nodes = G._nodes
            source_idx = G._index.get(source, -1)
            if source_idx == -1:
                raise nx.NodeNotFound(f"Source node {source} not in graph")
            
            raw_edges = _cpp_dfs_edges(graph=G._G, source=source_idx)
            edges = []
            for u_idx, v_idx in raw_edges:
                if u_idx != -1:
                    u, v = nodes[u_idx], nodes[v_idx]
                    edges.append((u, v))
            return iter(edges)
        elif isinstance(G, CppGraph):
            # if source < 0 or source >= len(G.edges()):
            if source < 0 or source >= G.n:
                raise nx.NodeNotFound(f"Source node {source} not in graph")
            
            raw_edges = _cpp_dfs_edges(graph=G, source=source)
            edges = []
            for u, v in raw_edges:
                if u != -1:
                    edges.append((u, v))
            return iter(edges)
        else:
            # return dfs_edges(convert_from_nx(G), source=source, depth_limit=depth_limit)
            # if dispatcher called this function, we will always get an NxCppGraph
            # if we reach here, that means the user directly called the nx_cpp.backend.function with a NetworkX graph, so we use the Python backend
            return nx.dfs_edges(G, source=source, depth_limit=depth_limit, sort_neighbors=sort_neighbors)
    except Exception:
        G = convert_to_nx(G)
        return nx.dfs_edges(G, source=source, depth_limit=depth_limit, sort_neighbors=sort_neighbors)


def shortest_path_pair(G, source, target, weight, method):
    """
    shortest path between source and target using C++ backend
    """
    nodes = G._nodes

    source_idx = G._index.get(source, -1)
    if source_idx == -1:
        raise nx.NodeNotFound(f"Source node {source} not in graph")

    target_idx = G._index.get(target, -1)
    if target_idx == -1:
        raise nx.NodeNotFound(f"Target node {target} not in graph")

    # choose algorithm
    if weight is None:
        # unweighted, ignore method
        distances, parent = _cpp_unweighted_shortest_path(graph=G._G, source=source_idx)
    elif weight == "weight":
        if method == "dijkstra":
            distances, parent = _cpp_dijkstra(graph=G._G, source=source_idx)
        else:
            distances, parent = _cpp_bellman_ford(graph=G._G, source=source_idx)
    else:
        raise ValueError("Unsupported weight attribute for C++ backend")

    if distances[target_idx] == float("inf"):
        raise nx.NetworkXNoPath(f"No path from {source} to {target}")

    # reconstruct path
    path_indices = []
    curr = target_idx
    while curr != -1:
        path_indices.append(curr)
        curr = parent[curr]
    path_indices.reverse()

    return [nodes[i] for i in path_indices]


def shortest_path(G, source=None, target=None, weight=None, method="dijkstra"):
    """
    Backend implementation for nx.shortest_path
    Only used for the (source, target) pair case; other arguments being used leads to Python fallback
    """
    if source is None or target is None or method not in ("dijkstra", "bellman-ford") or weight not in ("weight", None):
        G = convert_to_nx(G)
        return nx.shortest_path(
            G,
            source=source,
            target=target,
            weight=weight,
            method=method
        )

    try:
        if isinstance(G, NxCppGraph):
            return shortest_path_pair(
                G=G,
                source=source,
                target=target,
                weight=weight,
                method=method
            )
        else:
            return nx.shortest_path(
                G,
                source=source,
                target=target,
                weight=weight,
                method=method
            )
    except Exception:
        G = convert_to_nx(G)
        return nx.shortest_path(
            G,
            source=source,
            target=target,
            weight=weight,
            method=method
        )

def betweenness_centrality(
    G,
    k=None,
    normalized=True,
    weight=None,
    endpoints=False,
    seed=None,
    *args,
    **kwargs,
):
    """
    Backend implementation for nx.betweenness_centrality
    Uses Brandes' algorithm for unweighted graphs
    Only normalized and endpoints=False are used; other kwargs being used leads to Python fallback
    """
    if k is not None or weight is not None or endpoints:
        G = convert_to_nx(G)
        return nx.betweenness_centrality(
            G,
            k=k,
            normalized=normalized,
            weight=weight,
            endpoints=endpoints,
            seed=seed
        )
    try:
        if isinstance(G, NxCppGraph):
            bc = _cpp_betweenness_centrality(graph=G._G, normalized=normalized, endpoints=endpoints)
            return {node: float(bc[i]) for i, node in enumerate(G._nodes)}
        elif isinstance(G, CppGraph):
            bc = _cpp_betweenness_centrality(graph=G, normalized=normalized, endpoints=endpoints)
            return {i: float(bc[i]) for i in range(len(bc))}
        else:
            # return betweenness_centrality(convert_from_nx(G), k=k, normalized=normalized, weight=weight, endpoints=endpoints, seed=seed)
            # if dispatcher called this function, we will always get an NxCppGraph
            # if we reach here, that means the user directly called the nx_cpp.backend.function with a NetworkX graph, so we use the Python backend
            return nx.betweenness_centrality(
                G,
                k=k,
                normalized=normalized,
                weight=weight,
                endpoints=endpoints,
                seed=seed
            )
    except Exception:
        G = convert_to_nx(G)
        return nx.betweenness_centrality(
            G,
            k=k,
            normalized=normalized,
            weight=weight,
            endpoints=endpoints,
            seed=seed
        )


def connected_components(G, method="union-find", **kwargs):
    """
    Backend implementation for nx.connected_components
    Compute connected components using either union-find (default) or BFS
    Returns components sorted by size in descending order
    """
    method_normalized = method.lower().replace("_", "-")
    if method_normalized not in ("union-find", "bfs"):
        raise ValueError(f"Unknown method for connected_components: {method}")
    try:
        if isinstance(G, NxCppGraph):
            if G.is_directed():
                raise nx.NetworkXNotImplemented("connected_components is defined for undirected graphs")
            if method_normalized == "union-find":
                component_ids = _cpp_connected_components_union_find(graph=G._G)
            else:
                component_ids = _cpp_connected_components_bfs(graph=G._G)
            return _component_sets_from_ids(G._nodes, component_ids)
        return connected_components(convert_from_nx(G), method=method, **kwargs)
    except Exception:
        G = convert_to_nx(G)
        return nx.connected_components(G)


def connected_components_union_find(G, **kwargs):
    return connected_components(G, method="union-find", **kwargs)


def connected_components_bfs(G, **kwargs):
    return connected_components(G, method="bfs", **kwargs)


def minimum_spanning_tree(G, weight="weight", algorithm="kruskal", ignore_nan=False):
    """
    Backend implementation for nx.minimum_spanning_tree
    Uses Kruskal's or Prim's algorithm
    If ignore_nan is True or algorithm is Boruvka, falls back to Python NetworkX
    """
    if ignore_nan or (algorithm != "kruskal" and algorithm != "prim"):
        G = convert_to_nx(G)
        return nx.minimum_spanning_tree(G, weight=weight, algorithm=algorithm, ignore_nan=ignore_nan)
    try:
        algo = algorithm.lower()
        if algo not in ("kruskal", "prim"):
            raise ValueError(f"Unknown MST algorithm: {algorithm}")
        if isinstance(G, NxCppGraph):
            if G.is_directed():
                raise nx.NetworkXNotImplemented("minimum_spanning_tree requires an undirected graph")
            edges = _cpp_minimum_spanning_tree(graph=G._G, algorithm=algo)
            H = nx.Graph()
            H.add_nodes_from(G._nodes)
            for u_idx, v_idx, weight_val in edges:
                u = G._nodes[u_idx]
                v = G._nodes[v_idx]
                H.add_edge(u, v, weight=float(weight_val))
            return H
        else:
            G = convert_to_nx(G)
            return nx.minimum_spanning_tree(G, weight=weight, algorithm=algorithm, ignore_nan=ignore_nan)
            # return minimum_spanning_tree(
            #     convert_from_nx(G, weight=weight),
            #     weight=weight,
            #     algorithm=algo,
            #     ignore_nan=ignore_nan,
            # )
    except Exception:
        G = convert_to_nx(G)
        return nx.minimum_spanning_tree(G, weight=weight, algorithm=algorithm, ignore_nan=ignore_nan)

def is_isomorphic(G1, G2, node_match=None, edge_match=None, **kwargs):
    """
    Backend implementation for nx.is_isomorphic
    Exact graph isomorphism test using a backtracking search with heuristics
    If any match functions are used, falls back to Python NetworkX
    """
    if node_match is not None or edge_match is not None:
        G1 = convert_to_nx(G1)
        G2 = convert_to_nx(G2)
        return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=edge_match)
    try:
        H1 = G1 if isinstance(G1, NxCppGraph) else convert_from_nx(G1)
        H2 = G2 if isinstance(G2, NxCppGraph) else convert_from_nx(G2)
        if H1.is_directed() != H2.is_directed():
            raise nx.NetworkXError("Graphs must both be directed or undirected")
        return bool(_cpp_graphs_are_isomorphic(graph_a=H1._G, graph_b=H2._G))
    except Exception:
        G1 = convert_to_nx(G1)
        G2 = convert_to_nx(G2)
        return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=edge_match)


backend = sys.modules[__name__]


def get_info():
    return {
        "backend_name": "cpp",
        "project": "nx-cpp",
        "package": "nx-cpp",
        "url": "https://example.com/nx-cpp",  # TODO
        "short_summary": "Minimal C++ backend for NetworkX with a fast pagerank.",
        "default_config": {},
        "functions": {
            "pagerank": {
                "additional_docs": "Unweighted PageRank; ignores extra kwargs.",
                "additional_parameters": {
                    "alpha : float": "Damping factor (default 0.85).",
                    "max_iter : int": "Maximum iterations (default 100).",
                    "tol : float": "Convergence tolerance (default 1e-6).",
                },
            },
            "bfs_edges": {
                "additional_docs": "BFS traversal; ignores depth_limit and sort_neighbors.",
                "additional_parameters": {
                    "source": "Starting node for BFS traversal.",
                    "reverse : bool": "If True, yield edges in reverse direction (default False).",
                },
            },
            "dfs_edges": {
                "additional_docs": "DFS traversal; ignores depth_limit.",
                "additional_parameters": {
                    "source": "Starting node for DFS traversal.",
                },
            },
            "shortest_path": {
                "additional_docs": "Shortest path using Dijkstra or Bellman-Ford.",
                "additional_parameters": {
                    "source": "Starting node for path computation.",
                    "target": "Ending node (optional, returns dict if None).",
                    "weight : str": "Edge data key for weight (default 'weight').",
                    "method : str": "Algorithm: 'dijkstra' (default) or 'bellman-ford'.",
                },
            },
            "betweenness_centrality": {
                "additional_docs": "Betweenness centrality using Brandes' algorithm for unweighted graphs; k, weight, and seed parameters are ignored.",
                "additional_parameters": {
                    "normalized : bool": "If True, normalize by 2/((n-1)(n-2)) for undirected graphs (default True).",
                    "endpoints : bool": "If True, include endpoints in shortest path counts (default False).",
                },
            },
            "connected_components": {
                "additional_docs": "Connected components via union-find (default) or BFS.",
                "additional_parameters": {
                    "method : str": "Either 'union-find' (default) or 'bfs'.",
                },
            },
            "minimum_spanning_tree": {
                "additional_docs": "Minimum spanning forest using Kruskal or Prim.",
                "additional_parameters": {
                    "algorithm : str": "Either 'kruskal' (default) or 'prim'.",
                    "weight : str": "Edge data key for weight extraction during conversion (default 'weight').",
                    "ignore_nan : bool": "Ignored placeholder to mirror NetworkX API (default False).",
                },
            },
            "is_isomorphic": {
                "additional_docs": "Exact graph isomorphism via backtracking search with degree heuristics.",
                "additional_parameters": {},
            },
        },
    }
