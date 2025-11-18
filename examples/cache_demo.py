import time
import networkx as nx

from nx_cpp.backend import convert_from_nx, clear_nx_cpp_cache

"""Demonstrate conversion caching performance."""

def main():
    print("Creating graph...")
    G = nx.gnp_random_graph(20_000, 0.0008)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\nFirst conversion (warm build)...")
    t0 = time.time(); cg1 = convert_from_nx(G); t1 = time.time()
    print(f"Time: {t1 - t0:.3f}s")

    print("Second conversion (should hit cache)...")
    t2 = time.time(); cg2 = convert_from_nx(G); t3 = time.time()
    print(f"Time: {t3 - t2:.3f}s (expected ~ near-zero)")
    print("Same object id:", id(cg1) == id(cg2))

    print("\nForced rebuild (use_cache=False)...")
    t4 = time.time(); cg3 = convert_from_nx(G, use_cache=False); t5 = time.time()
    print(f"Time: {t5 - t4:.3f}s (similar to first)")
    print("New object id:", id(cg3) != id(cg2))

    print("\nClear cache then convert again...")
    clear_nx_cpp_cache(G)
    t6 = time.time(); cg4 = convert_from_nx(G); t7 = time.time()
    print(f"Time: {t7 - t6:.3f}s (back to full build)")

    # Demonstrate backend algorithm call timings with implicit conversion + cache reuse
    print("\n=== Algorithm timing with implicit conversion cache ===")
    # Clear cache to force conversion inside first pagerank call
    clear_nx_cpp_cache(G)
    print("Running pagerank first time (will convert)...")
    a0 = time.time(); pr1 = nx.pagerank(G, backend='cpp'); a1 = time.time()
    print(f"First pagerank time (includes conversion): {a1 - a0:.3f}s")

    print("Running pagerank second time (cache hit, no reconversion)...")
    a2 = time.time(); pr2 = nx.pagerank(G, backend='cpp'); a3 = time.time()
    print(f"Second pagerank time: {a3 - a2:.3f}s")
    # Basic sanity: top node matches
    top1 = max(pr1.items(), key=lambda x: x[1])[0]
    top2 = max(pr2.items(), key=lambda x: x[1])[0]
    print("Top node stable:", top1 == top2)

    print("Clearing cache and re-running pagerank (forces reconversion)...")
    clear_nx_cpp_cache(G)
    a4 = time.time(); pr3 = nx.pagerank(G, backend='cpp'); a5 = time.time()
    print(f"After clear, pagerank time: {a5 - a4:.3f}s (should resemble first)")

if __name__ == '__main__':
    main()
