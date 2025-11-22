# nx-cpp

A NetworkX backend that implements common functions in C++ via pybind11.

## Install

With Python venv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

With Conda:
```bash
conda create -n nx-cpp pip
conda activate nx-cpp
pip install -e .
```

You may need to install `libomp` before building the project. On Apple Silicon with Homebrew, this can be achieved with:
```bash
brew install libomp
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"
```

## Try it

```python
import networkx as nx

G = nx.gnp_random_graph(1000, 0.01, directed=True)

# Basic NetworkX
pr_py = nx.pagerank(G, alpha=0.85)

# NetworkX w/ C++ backend
pr_cpp = nx.pagerank(G, alpha=0.85, backend="cpp")

# Auto-dispatch (optional)
# import os; os.environ["NETWORKX_BACKEND_PRIORITY"] = "cpp"
```

## Important Notice

Always invoke algorithms through the official NetworkX API; bypassing it can cause the C++ backend to operate on outdated cached data.

## Functionality

Functions supported:
- `pagerank(G, alpha, max_iter, weight, tol)`
- `bfs_edges(G, source)`
- `dfs_edges(G, source)`
- `shortest_path(G, source, target, weight, method)`
- `betweenness_centrality(G)`
- `connected_components(G, method)`
- `minimum_spanning_tree(G, weight, algorithm)`
- `is_isomorphic(G1, G2)`

## Limitations

Multigraphs are not supported.

## Best Uses

Most effective on medium to large graphs, where algorithmic speedups outweigh dispatch and conversion overhead.

Conversion cost overhead can sometimes make C++ backend slower than Python on small graphs doing lightweight computations.

## Repository Structure

`nx_cpp/`: source code for the backend, including C++ definitions, python bindings, and backend dispatching.

`examples/`: short, lightweight demos for each of the implemented algorithms that highlight uses, speedup, and accuracy

`tests/`: comprehensive test suite covering unit tests, large-graph evaluations, memory checks, and end-to-end scenarios emulating real-world usage.

## How to Test

In the project root, run:
```bash
python -m pytest -vv -s
```
`-vv` and `s` will print out the actual runtimes and speedups for individual stress tests.

`pytest.ini` contains a list of all the markers. Subsets of tests can be run in isolation.

## Hardware Requirements

Apple Silicon and Linux (school VM) both support the execution of this program.
