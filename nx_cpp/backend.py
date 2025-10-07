import sys
import networkx as nx
import time

# pybind11 module
from ._nx_cpp import Graph as CppGraph, pagerank as _cpp_pagerank


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


def convert_from_nx(G, **kwargs):
    """
    Convert a NetworkX Graph -> NxCppGraph.
    - Nodes are relabeled to 0..n-1 for C++
    - Parallel edges collapsed (no multigraph support)
    """
    nodes = list(G.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    directed = G.is_directed()

    if G.is_multigraph():
        edges = {(index[u], index[v]) for (u, v, _k) in G.edges(keys=True)}
        edges = list(edges)
    else:
        edges = [(index[u], index[v]) for (u, v) in G.edges()]

    cpp_graph = CppGraph(len(nodes), edges, directed)
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
    """We only implement pagerank."""
    return name == "pagerank"


def should_run(name, args, kwargs):
    """Prefer C++ when the graph is not tiny (heuristic)."""
    if name != "pagerank":
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
            }
        },
    }
