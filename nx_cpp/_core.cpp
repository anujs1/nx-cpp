#include <cmath>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>
#include <vector>

namespace py = pybind11;

struct Graph {
  int n;
  std::vector<std::vector<int>> out_adj;
  bool directed;

  Graph(int n_, const std::vector<std::pair<int, int>> &edges, bool directed_)
      : n(n_), out_adj(n_), directed(directed_) {
    for (const auto &e : edges) {
      int u = e.first, v = e.second;
      if (u < 0 || u >= n || v < 0 || v >= n)
        continue;
      out_adj[u].push_back(v);
      if (!directed) {
        out_adj[v].push_back(u);
      }
    }
  }

  bool is_directed() const { return directed; }
  bool is_multigraph() const { return false; }

  std::vector<std::pair<int, int>> edges() const {
    std::vector<std::pair<int, int>> E;
    for (int u = 0; u < n; ++u) {
      for (int v : out_adj[u])
        E.emplace_back(u, v);
    }
    return E;
  }
};

std::vector<double> pagerank(const Graph &G, double alpha, int max_iter,
                             double tol) {
  const int n = G.n;
  if (n == 0)
    return {};
  std::vector<double> pr(n, 1.0 / n);
  std::vector<double> next(n, 0.0);
  std::vector<int> outdeg(n, 0);
  for (int u = 0; u < n; ++u)
    outdeg[u] = static_cast<int>(G.out_adj[u].size());

  for (int it = 0; it < max_iter; ++it) {
    double dangling_sum = 0.0;
    for (int u = 0; u < n; ++u)
      if (outdeg[u] == 0)
        dangling_sum += pr[u];

    const double base = (1.0 - alpha) / n;
    const double dang = alpha * dangling_sum / n;
    for (int i = 0; i < n; ++i)
      next[i] = base + dang;

    for (int u = 0; u < n; ++u) {
      if (outdeg[u] == 0)
        continue;
      const double share = alpha * pr[u] / outdeg[u];
      for (int v : G.out_adj[u])
        next[v] += share;
    }

    double diff = 0.0;
    for (int i = 0; i < n; ++i)
      diff += std::abs(next[i] - pr[i]);

    // normalizing pr to ensure sum stays at 1
    double s = std::accumulate(next.begin(), next.end(), 0.0);
    if (s > 0)
      for (double &x : next)
        x /= s;
    pr.swap(next);

    // adding a normalization to diff to avoid exiting too early
    if (diff / n < tol)
      break;
  }
  return pr;
}

PYBIND11_MODULE(_nx_cpp, m) {
  auto graph_cls = py::class_<Graph>(m, "Graph");
  graph_cls
      .def(py::init<int, const std::vector<std::pair<int, int>> &, bool>(),
           py::arg("num_nodes"), py::arg("edges"), py::arg("directed") = false)
      .def("is_directed", &Graph::is_directed)
      .def("is_multigraph", &Graph::is_multigraph)
      .def("edges", &Graph::edges);
  graph_cls.attr("__networkx_backend__") = "cpp";

  m.def("pagerank", &pagerank, py::arg("graph"), py::arg("alpha") = 0.85,
        py::arg("max_iter") = 100, py::arg("tol") = 1e-6,
        "Minimal unweighted PageRank implementation.");
}
