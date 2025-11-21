import networkx as nx

def run_pagerank():
    G = nx.gnp_random_graph(100, 0.05, seed=0, directed=True)
    pr = nx.pagerank(G, alpha=0.85, backend="cpp")
    print("[valgrind] pagerank ran; nodes:", len(pr))


def run_bfs_edges():
    G = nx.balanced_tree(r=2, h=5, create_using=nx.DiGraph)
    source = 0
    edges = list(nx.bfs_edges(G, source, backend="cpp"))
    print("[valgrind] bfs_edges ran; edges:", len(edges))


def run_dfs_edges():
    G = nx.balanced_tree(r=2, h=5, create_using=nx.DiGraph)
    source = 0
    edges = list(nx.dfs_edges(G, source, backend="cpp"))
    print("[valgrind] dfs_edges ran; edges:", len(edges))


def run_shortest_path():
    G = nx.gnp_random_graph(80, 0.05, seed=1)
    for u, v in G.edges:
        G[u][v]["weight"] = 1.0
    source = 0
    target = max(G.nodes)
    path = nx.shortest_path(G, source=source, target=target, weight="weight", backend="cpp")
    print("[valgrind] shortest_path ran; path length:", len(path))


def run_betweenness():
    G = nx.gnp_random_graph(60, 0.08, seed=2)
    bc = nx.betweenness_centrality(G, backend="cpp")
    print("[valgrind] betweenness_centrality ran; nodes:", len(bc))


def run_connected_components():
    G = nx.Graph()
    comp1 = [0, 1, 2, 3, 4]
    G.add_edges_from(zip(comp1, comp1[1:]))
    comp2 = [10, 11, 12, 13]
    G.add_edges_from(zip(comp2, comp2[1:]))
    G.add_node(20)

    comps = list(nx.connected_components(G, backend="cpp"))
    print("[valgrind] connected_components ran; components:", len(comps))


def run_mst():
    G = nx.gnp_random_graph(50, 0.1, seed=3)
    for u, v in G.edges:
        G[u][v]["weight"] = 1.0

    T = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal", backend="cpp")
    print("[valgrind] minimum_spanning_tree ran; mst edges:", len(T.edges))


def run_is_isomorphic():
    G1 = nx.gnp_random_graph(30, 0.2, seed=4)
    nodes = list(G1.nodes)
    mapping = {u: v for u, v in zip(nodes, reversed(nodes))}
    G2 = nx.relabel_nodes(G1, mapping)

    iso = nx.is_isomorphic(G1, G2, backend="cpp")
    print("[valgrind] is_isomorphic ran; result:", iso)


def main():
    run_pagerank()
    run_bfs_edges()
    run_dfs_edges()
    run_shortest_path()
    run_betweenness()
    run_connected_components()
    run_mst()
    run_is_isomorphic()


if __name__ == "__main__":
    main()
