# nx-cpp

A tiny NetworkX backend that implements `pagerank` in C++ via pybind11.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
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

Notes:

- Weighted and/or multigraphs are not supported.
- `convert_to_nx` drops attributes.
