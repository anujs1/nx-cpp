# from .backend import (
#     NxCppGraph,
#     backend,
#     convert_from_nx,
#     convert_to_nx,
#     get_info,
#     is_isomorphic,
#     minimum_spanning_tree,
#     pagerank,
#     connected_components,
#     connected_components_union_find,
#     connected_components_bfs,
#     bfs_edges,
#     dfs_edges,
#     shortest_path,
#     betweenness_centrality,
# )

from importlib import import_module

__all__ = [
    "NxCppGraph",
    "backend",
    "convert_from_nx",
    "convert_to_nx",
    "get_info",
    "is_isomorphic",
    "minimum_spanning_tree",
    "pagerank",
    "connected_components",
    "connected_components_union_find",
    "connected_components_bfs",
    "bfs_edges",
    "dfs_edges",
    "shortest_path",
    "betweenness_centrality",
]

def __getattr__(name):
    """
    lazily expose functions from nx_cpp.backend to avoid circular import error
    """
    if name in __all__:
        backend = import_module("nx_cpp.backend")
        return getattr(backend, name)
    raise AttributeError(f"module 'nx_cpp' has no attribute {name!r}")