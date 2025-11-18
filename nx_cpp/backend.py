import sys
import networkx as nx
import time

# pybind11 module
from ._nx_cpp import (
    Graph as CppGraph,
    pagerank as _cpp_pagerank,
    bfs_edges as _cpp_bfs_edges,
    dfs_edges as _cpp_dfs_edges,
    dijkstra as _cpp_dijkstra,
    bellman_ford as _cpp_bellman_ford,
    betweenness_centrality as _cpp_betweenness_centrality,
)


class NxCppGraph:
    """
    Minimal backend graph wrapper understood by NetworkX dispatch.
    Holds a C++ Graph and preserves original Python node labels.
    """

    __networkx_backend__ = "cpp"

    def __init__(self, cpp_graph: CppGraph, nodes):
        self._G = cpp_graph
        self._nodes = list(nodes)

    def is_directed(self) -> bool:
        return self._G.is_directed()

    def is_multigraph(self) -> bool:
        return self._G.is_multigraph()

    def _edges_py(self):
        idx_edges = self._G.edges()
        nodes = self._nodes
        return [(nodes[u], nodes[v]) for (u, v) in idx_edges]


def convert_from_nx(G, weight='weight', **kwargs):
    """
    Convert a NetworkX Graph -> NxCppGraph.
    - Nodes are relabeled to 0..n-1 for C++
    - Parallel edges collapsed (no multigraph support)
    - Supports weighted graphs via weight parameter
    """
    t_start = time.time()
    nodes = list(G.nodes())
    t_nodes = time.time()
    
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
    
    t_edges = time.time()
    cpp_graph = CppGraph(len(nodes), edges, directed)
    t_cpp = time.time()
    
    print(f"  [Conversion] nodes: {t_nodes - t_start:.3f}s, edges: {t_edges - t_nodes:.3f}s, cpp_graph: {t_cpp - t_edges:.3f}s, total: {t_cpp - t_start:.3f}s")
    
    return NxCppGraph(cpp_graph, nodes)


def convert_to_nx(obj, **kwargs):
    """NxCppGraph -> NetworkX Graph/DiGraph (no attributes)."""
    if isinstance(obj, NxCppGraph):
        H = nx.DiGraph() if obj.is_directed() else nx.Graph()
        H.add_nodes_from(obj._nodes)
        H.add_edges_from(obj._edges_py())
        return H
    return obj


def can_run(name, args, kwargs):
    """We implement pagerank, bfs_edges, dfs_edges, shortest_path, and betweenness_centrality functions."""
    return name in ("pagerank", "bfs_edges", "dfs_edges", "shortest_path", "dijkstra_path", "bellman_ford_path", "betweenness_centrality")


def should_run(name, args, kwargs):
    """Prefer C++ when the graph is not tiny (heuristic)."""
    if name not in ("pagerank", "bfs_edges", "dfs_edges", "shortest_path", "dijkstra_path", "bellman_ford_path", "betweenness_centrality"):
        return False
    G = args[0] if args else kwargs.get("G", None)
    try:
        n = len(G._nodes) if isinstance(G, NxCppGraph) else G.number_of_nodes()
    except Exception:
        n = 0
    return n >= 50


def pagerank(
    G,
    alpha: float = 0.85,
    personalization=None,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    nstart=None,
    weight=None,
    dangling=None,
    *args,
    **kwargs,
):
    """
    Backend implementation for nx.pagerank.
    Only alpha, max_iter, tol are used; other kwargs ignored.
    """
    if isinstance(G, NxCppGraph):
        pr = _cpp_pagerank(graph=G._G, alpha=alpha, max_iter=max_iter, tol=tol)
        return {node: float(pr[i]) for i, node in enumerate(G._nodes)}
    elif isinstance(G, CppGraph):
        pr = _cpp_pagerank(graph=G, alpha=alpha, max_iter=max_iter, tol=tol)
        return {i: float(pr[i]) for i in range(len(pr))}
    else:
        return pagerank(convert_from_nx(G), alpha=alpha, max_iter=max_iter, tol=tol)


def bfs_edges(G, source, reverse=False, depth_limit=None, sort_neighbors=None):
    """
    Backend implementation for nx.bfs_edges.
    Returns an iterator of edges in BFS order from source.
    Only basic BFS is supported; depth_limit and sort_neighbors are ignored.
    """
    if isinstance(G, NxCppGraph):
        nodes = G._nodes
        try:
            source_idx = nodes.index(source)
        except ValueError:
            raise nx.NodeNotFound(f"Source node {source} not in graph")
        
        parent = _cpp_bfs_edges(graph=G._G, source=source_idx)
        edges = []
        for v_idx, u_idx in enumerate(parent):
            if u_idx != -1:
                u, v = nodes[u_idx], nodes[v_idx]
                edges.append((u, v) if not reverse else (v, u))
        return iter(edges)
    elif isinstance(G, CppGraph):
        if source < 0 or source >= len(G.edges()):
            raise nx.NodeNotFound(f"Source node {source} not in graph")
        
        parent = _cpp_bfs_edges(graph=G, source=source)
        edges = []
        for v, u in enumerate(parent):
            if u != -1:
                edges.append((u, v) if not reverse else (v, u))
        return iter(edges)
    else:
        return bfs_edges(convert_from_nx(G), source=source, reverse=reverse, 
                        depth_limit=depth_limit, sort_neighbors=sort_neighbors)


def dfs_edges(G, source, depth_limit=None):
    """
    Backend implementation for nx.dfs_edges.
    Returns an iterator of edges in DFS order from source.
    Only basic DFS is supported; depth_limit is ignored.
    """
    if isinstance(G, NxCppGraph):
        nodes = G._nodes
        try:
            source_idx = nodes.index(source)
        except ValueError:
            raise nx.NodeNotFound(f"Source node {source} not in graph")
        
        parent = _cpp_dfs_edges(graph=G._G, source=source_idx)
        edges = []
        for v_idx, u_idx in enumerate(parent):
            if u_idx != -1:
                u, v = nodes[u_idx], nodes[v_idx]
                edges.append((u, v))
        return iter(edges)
    elif isinstance(G, CppGraph):
        if source < 0 or source >= len(G.edges()):
            raise nx.NodeNotFound(f"Source node {source} not in graph")
        
        parent = _cpp_dfs_edges(graph=G, source=source)
        edges = []
        for v, u in enumerate(parent):
            if u != -1:
                edges.append((u, v))
        return iter(edges)
    else:
        return dfs_edges(convert_from_nx(G), source=source, depth_limit=depth_limit)


def shortest_path(G, source, target=None, weight='weight', method='dijkstra'):
    """
    Backend implementation for nx.shortest_path.
    Supports Dijkstra's algorithm (default) and Bellman-Ford.
    Returns the shortest path from source to target, or a dict of paths to all nodes if target is None.
    """
    if isinstance(G, NxCppGraph):
        nodes = G._nodes
        try:
            source_idx = nodes.index(source)
        except ValueError:
            raise nx.NodeNotFound(f"Source node {source} not in graph")
        
        # Choose algorithm
        t_algo_start = time.time()
        if method == 'dijkstra':
            distances, parent = _cpp_dijkstra(graph=G._G, source=source_idx)
        elif method == 'bellman-ford':
            distances, parent = _cpp_bellman_ford(graph=G._G, source=source_idx)
        else:
            raise ValueError(f"Unknown method: {method}")
        t_algo_end = time.time()
        print(f"  [C++ {method}] algorithm time: {t_algo_end - t_algo_start:.3f}s")
        
        # Reconstruct paths
        t_recon_start = time.time()
        def get_path(target_idx):
            if distances[target_idx] == float('inf'):
                raise nx.NetworkXNoPath(f"No path from {source} to {nodes[target_idx]}")
            path = []
            current = target_idx
            while current != -1:
                path.append(nodes[current])
                current = parent[current]
            return list(reversed(path))
        
        if target is not None:
            try:
                target_idx = nodes.index(target)
            except ValueError:
                raise nx.NodeNotFound(f"Target node {target} not in graph")
            result = get_path(target_idx)
        else:
            # Return dict of paths to all reachable nodes
            paths = {}
            for i, node in enumerate(nodes):
                if distances[i] != float('inf'):
                    paths[node] = get_path(i)
            result = paths
        t_recon_end = time.time()
        print(f"  [Python] path reconstruction time: {t_recon_end - t_recon_start:.3f}s")
        return result
    else:
        return shortest_path(convert_from_nx(G, weight=weight), source=source, target=target, weight=weight, method=method)


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
    Backend implementation for nx.betweenness_centrality.
    Uses Brandes' algorithm for unweighted graphs.
    Parameters k, weight, and seed are ignored in this implementation.
    """
    if isinstance(G, NxCppGraph):
        bc = _cpp_betweenness_centrality(graph=G._G, normalized=normalized, endpoints=endpoints)
        return {node: float(bc[i]) for i, node in enumerate(G._nodes)}
    elif isinstance(G, CppGraph):
        bc = _cpp_betweenness_centrality(graph=G, normalized=normalized, endpoints=endpoints)
        return {i: float(bc[i]) for i in range(len(bc))}
    else:
        return betweenness_centrality(convert_from_nx(G), k=k, normalized=normalized, weight=weight, endpoints=endpoints, seed=seed)


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
            }
        },
    }
