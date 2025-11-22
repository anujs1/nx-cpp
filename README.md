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

Use official NetworkX API function calls, or else the C++ backend will run on a stale cache

## Functionality

`pagerank`
`bfs_edges`
`dfs_edges`
`shortest_path`
`betweenness_centrality`
`connected_components`
`minimum_spanning_tree`
`is_isomorphic`

## Limitations

Multigraphs are not supported

## Best Uses

Works well on most graph types, true impact of speedup is most noticeable on large graphs

Conversion cost and overhead can sometimes make C++ backend slower than Python on small graphs doing inexpensive computations

## Repository Structure

`nx_cpp`
- contains all of the actual code for the backend, including C++ definitions, python bindings, and backend handling

`examples`
- contains short, lightweight demos for each of the implemented algorithms that highlight uses, speedup, and accuracy

`tests`
- comprehensive test suite that tests each function in isolation on small unit tests to large graph tests. also contains valgrind test and a real-world test that tests functions on programs simulating more "real-world" behavior and using functions together

## How to Test

From the root, can run `python -m pytest -vv -s`
Recommend the `-vv` and `-s` flags as they will print out actual runtimes and speedups for individual stress tests
`pytest.ini` contains a list of all the marks – specific tests can be run in isolation or avoided completely based on whatever you'd like to test

## Hardware Requirements

M1 Mac and Linux VM have both been tested with this program and should run cleanly.
