import networkx as nx

def load_dimacs_graph(path: str, directed=True):
    """
    Load a DIMACS-format road network file.
    Example line: 'a 1 2 3' means edge 1->2 weight=3
    """
    G = nx.DiGraph() if directed else nx.Graph()
    with open(path, "r") as f:
        for line in f:
            if not line or line.startswith("c") or line.startswith("p"):
                continue
            parts = line.strip().split()
            if parts[0] == "a" and len(parts) >= 4:
                u, v, w = int(parts[1])-1, int(parts[2])-1, float(parts[3])
                G.add_edge(u, v, weight=w)
    return G
