import networkx as nx
import time

"""Demonstrate backend with string-labeled nodes."""

def main():
    print("Creating string-labeled graph...")
    G = nx.Graph()
    # Create a simple clustered graph with string labels
    for i in range(50):
        G.add_node(f"user_{i}")
    # Add edges in two clusters plus bridges
    for i in range(0, 25):
        for j in range(i + 1, 25):
            if (i + j) % 11 == 0:
                G.add_edge(f"user_{i}", f"user_{j}")
    for i in range(25, 50):
        for j in range(i + 1, 50):
            if (i * j) % 17 == 0:
                G.add_edge(f"user_{i}", f"user_{j}")
    # Bridge edges
    for k in range(0, 25, 5):
        G.add_edge(f"user_{k}", f"user_{k+25}")

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    source = "user_0"
    print(f"\nBFS from {source} (Python)...")
    t0 = time.time(); edges_py = list(nx.bfs_edges(G, source)); t1 = time.time()
    print(f"Python time: {t1 - t0:.4f}s, edges: {len(edges_py)}")

    print(f"BFS from {source} (C++ backend)...")
    t2 = time.time(); edges_cpp = list(nx.bfs_edges(G, source, backend='cpp')); t3 = time.time()
    print(f"C++ time: {t3 - t2:.4f}s, edges: {len(edges_cpp)}")
    print("Match edge count:", len(edges_py) == len(edges_cpp))

    print(f"\nBetweenness centrality (subset) normalized comparison...")
    t4 = time.time(); bc_py = nx.betweenness_centrality(G); t5 = time.time()
    t6 = time.time(); bc_cpp = nx.betweenness_centrality(G, backend='cpp'); t7 = time.time()
    print(f"Python bc time: {t5 - t4:.4f}s; C++ bc time: {t7 - t6:.4f}s")
    # Show a few sample nodes
    for label in ["user_0", "user_10", "user_25", "user_30"]:
        print(f" {label}: py={bc_py[label]:.6f} cpp={bc_cpp[label]:.6f}")

if __name__ == '__main__':
    main()
