#include <cmath>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>
#include <vector>
#include <queue>
#include <stack>
#include <limits>
#include <algorithm>

namespace py = pybind11;

struct Graph {
  int n;
  std::vector<std::vector<int>> out_adj;
  std::vector<std::vector<double>> weights;  // weights[u][i] = weight of edge to out_adj[u][i]
  bool directed;
  bool weighted;

  Graph(int n_, const std::vector<std::pair<int, int>> &edges, bool directed_)
      : n(n_), out_adj(n_), weights(n_), directed(directed_), weighted(false) {
    for (const auto &e : edges) {
      int u = e.first, v = e.second;
      if (u < 0 || u >= n || v < 0 || v >= n)
        continue;
      out_adj[u].push_back(v);
      weights[u].push_back(1.0);  // default weight
      if (!directed) {
        out_adj[v].push_back(u);
        weights[v].push_back(1.0);
      }
    }
  }
  
  // Constructor with weights
  Graph(int n_, const std::vector<std::tuple<int, int, double>> &edges, bool directed_)
      : n(n_), out_adj(n_), weights(n_), directed(directed_), weighted(true) {
    for (const auto &e : edges) {
      int u = std::get<0>(e);
      int v = std::get<1>(e);
      double w = std::get<2>(e);
      if (u < 0 || u >= n || v < 0 || v >= n)
        continue;
      out_adj[u].push_back(v);
      weights[u].push_back(w);
      if (!directed) {
        out_adj[v].push_back(u);
        weights[v].push_back(w);
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

std::vector<int> bfs_edges(const Graph &G, int source) {
  const int n = G.n;
  if (source < 0 || source >= n)
    return {};
  
  std::vector<int> parent(n, -1);
  std::vector<bool> visited(n, false);
  std::queue<int> q;
  
  visited[source] = true;
  q.push(source);
  
  while (!q.empty()) {
    int u = q.front();
    q.pop();
    
    for (int v : G.out_adj[u]) {
      if (!visited[v]) {
        visited[v] = true;
        parent[v] = u;
        q.push(v);
      }
    }
  }
  
  return parent;
}

std::vector<int> dfs_edges(const Graph &G, int source) {
  const int n = G.n;
  if (source < 0 || source >= n)
    return {};
  
  std::vector<int> parent(n, -1);
  std::vector<bool> visited(n, false);
  std::stack<int> s;
  
  visited[source] = true;
  s.push(source);
  
  while (!s.empty()) {
    int u = s.top();
    s.pop();
    
    for (int v : G.out_adj[u]) {
      if (!visited[v]) {
        visited[v] = true;
        parent[v] = u;
        s.push(v);
      }
    }
  }
  
  return parent;
}

// Dijkstra's algorithm for shortest paths
std::pair<std::vector<double>, std::vector<int>> dijkstra(const Graph &G, int source) {
  const int n = G.n;
  const double INF = std::numeric_limits<double>::infinity();
  
  std::vector<double> dist(n, INF);
  std::vector<int> parent(n, -1);
  
  if (source < 0 || source >= n)
    return {dist, parent};
  
  // Priority queue: (distance, node)
  using pdi = std::pair<double, int>;
  std::priority_queue<pdi, std::vector<pdi>, std::greater<pdi>> pq;
  
  dist[source] = 0.0;
  pq.push({0.0, source});
  
  while (!pq.empty()) {
    auto [d, u] = pq.top();
    pq.pop();
    
    if (d > dist[u])
      continue;
    
    for (size_t i = 0; i < G.out_adj[u].size(); ++i) {
      int v = G.out_adj[u][i];
      double w = G.weights[u][i];
      double new_dist = dist[u] + w;
      
      if (new_dist < dist[v]) {
        dist[v] = new_dist;
        parent[v] = u;
        pq.push({new_dist, v});
      }
    }
  }
  
  return {dist, parent};
}

// Bellman-Ford algorithm for shortest paths (handles negative weights)
std::pair<std::vector<double>, std::vector<int>> bellman_ford(const Graph &G, int source) {
  const int n = G.n;
  const double INF = std::numeric_limits<double>::infinity();
  
  std::vector<double> dist(n, INF);
  std::vector<int> parent(n, -1);
  
  if (source < 0 || source >= n)
    return {dist, parent};
  
  dist[source] = 0.0;
  
  // Relax edges n-1 times
  for (int iter = 0; iter < n - 1; ++iter) {
    bool updated = false;
    for (int u = 0; u < n; ++u) {
      if (dist[u] == INF)
        continue;
      
      for (size_t i = 0; i < G.out_adj[u].size(); ++i) {
        int v = G.out_adj[u][i];
        double w = G.weights[u][i];
        double new_dist = dist[u] + w;
        
        if (new_dist < dist[v]) {
          dist[v] = new_dist;
          parent[v] = u;
          updated = true;
        }
      }
    }
    if (!updated)
      break;
  }
  
  return {dist, parent};
}

// Betweenness centrality using Brandes' algorithm
std::vector<double> betweenness_centrality(const Graph &G, bool normalized, bool endpoints) {
  const int n = G.n;
  std::vector<double> bc(n, 0.0);
  
  if (n == 0)
    return bc;
  
  // For each source node
  for (int s = 0; s < n; ++s) {
    std::stack<int> S;
    std::vector<std::vector<int>> P(n);  // predecessors
    std::vector<int> sigma(n, 0);        // number of shortest paths
    std::vector<int> dist(n, -1);        // distance from source
    std::vector<double> delta(n, 0.0);   // dependency
    
    sigma[s] = 1;
    dist[s] = 0;
    
    // BFS
    std::queue<int> Q;
    Q.push(s);
    
    while (!Q.empty()) {
      int v = Q.front();
      Q.pop();
      S.push(v);
      
      for (int w : G.out_adj[v]) {
        // First time visiting w?
        if (dist[w] < 0) {
          Q.push(w);
          dist[w] = dist[v] + 1;
        }
        // Shortest path to w via v?
        if (dist[w] == dist[v] + 1) {
          sigma[w] += sigma[v];
          P[w].push_back(v);
        }
      }
    }
    
    // Accumulation phase
    while (!S.empty()) {
      int w = S.top();
      S.pop();
      
      for (int v : P[w]) {
        delta[v] += (static_cast<double>(sigma[v]) / sigma[w]) * (1.0 + delta[w]);
      }
      
      if (w != s) {
        bc[w] += delta[w];
      }
    }
  }
  
  // Normalization / scaling to match NetworkX
  if (normalized && n > 2) {
    // For undirected graphs, NetworkX normalized value equals
    // (unnormalized/2) * (2/((n-1)(n-2))) = unnormalized * (1/((n-1)(n-2)))
    double scale = 1.0 / ((n - 1) * (n - 2));
    for (int i = 0; i < n; ++i)
      bc[i] *= scale;
  } else if (!G.directed) {
    // Unnormalized undirected values are divided by 2
    for (int i = 0; i < n; ++i)
      bc[i] /= 2.0;
  }
  
  // Handle endpoints
  if (endpoints) {
    // The standard Brandes algorithm doesn't count endpoints
    // If endpoints=True, we add them back (not implemented in this basic version)
  }
  
  return bc;
}

PYBIND11_MODULE(_nx_cpp, m) {
  auto graph_cls = py::class_<Graph>(m, "Graph");
  graph_cls
      .def(py::init<int, const std::vector<std::pair<int, int>> &, bool>(),
           py::arg("num_nodes"), py::arg("edges"), py::arg("directed") = false)
      .def(py::init<int, const std::vector<std::tuple<int, int, double>> &, bool>(),
           py::arg("num_nodes"), py::arg("edges_weighted"), py::arg("directed") = false)
      .def("is_directed", &Graph::is_directed)
      .def("is_multigraph", &Graph::is_multigraph)
      .def("edges", &Graph::edges);
  graph_cls.attr("__networkx_backend__") = "cpp";

  m.def("pagerank", &pagerank, py::arg("graph"), py::arg("alpha") = 0.85,
        py::arg("max_iter") = 100, py::arg("tol") = 1e-6,
        "Minimal unweighted PageRank implementation.");
  
  m.def("bfs_edges", &bfs_edges, py::arg("graph"), py::arg("source"),
        "BFS traversal returning parent array for edge reconstruction.");
  
  m.def("dfs_edges", &dfs_edges, py::arg("graph"), py::arg("source"),
        "DFS traversal returning parent array for edge reconstruction.");
  
  m.def("dijkstra", &dijkstra, py::arg("graph"), py::arg("source"),
        "Dijkstra's algorithm returning distances and parent array.");
  
  m.def("bellman_ford", &bellman_ford, py::arg("graph"), py::arg("source"),
        "Bellman-Ford algorithm returning distances and parent array.");
  
  m.def("betweenness_centrality", &betweenness_centrality, 
        py::arg("graph"), py::arg("normalized") = true, py::arg("endpoints") = false,
        "Betweenness centrality using Brandes' algorithm.");
}
