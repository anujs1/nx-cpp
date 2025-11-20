import networkx as nx
import nx_cpp


def describe(result):
    return "isomorphic" if result else "non-isomorphic"


def main():
    print("=== Graph Isomorphism Demo ===")

    base = nx.cycle_graph(6)
    G1 = base.copy()
    mapping = {i: (i + 3) % 6 for i in base.nodes()}
    G2 = nx.relabel_nodes(base, mapping)

    H = nx.path_graph(6)

    print("G1 vs G2 (should be isomorphic)")
    print("NetworkX:", describe(nx.is_isomorphic(G1, G2)))
    print("nx-cpp  :", describe(nx_cpp.is_isomorphic(G1, G2)))

    print("\nG1 vs H (should not be isomorphic)")
    print("NetworkX:", describe(nx.is_isomorphic(G1, H)))
    print("nx-cpp  :", describe(nx_cpp.is_isomorphic(G1, H)))

    print("\nRandom relabel sanity check")
    rnd = nx.gnp_random_graph(200, 0.02, seed=3)
    relabeled = nx.relabel_nodes(rnd, {u: u * 3 % 200 for u in rnd})
    result = nx_cpp.is_isomorphic(rnd, relabeled)
    print("nx-cpp result:", describe(result))


if __name__ == "__main__":
    main()
