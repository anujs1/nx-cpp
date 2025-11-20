#include <cmath>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>
#include <tuple>
#include <vector>
#include <queue>
#include <stack>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>

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

struct DisjointSet {
  std::vector<int> parent;
  std::vector<int> rank;

  explicit DisjointSet(int n) : parent(n), rank(n, 0) {
    std::iota(parent.begin(), parent.end(), 0);
  }

  int find(int x) {
    if (parent[x] != x)
      parent[x] = find(parent[x]);
    return parent[x];
  }

  void unite(int x, int y) {
    int rx = find(x);
    int ry = find(y);
    if (rx == ry)
      return;
    if (rank[rx] < rank[ry]) {
      parent[rx] = ry;
    } else if (rank[rx] > rank[ry]) {
      parent[ry] = rx;
    } else {
      parent[ry] = rx;
      rank[rx]++;
    }
  }
};

void require_undirected(const Graph &G, const std::string &context) {
  if (G.directed) {
    throw std::invalid_argument(context + " requires an undirected graph");
  }
}

struct WeightedEdge {
  int u;
  int v;
  double w;
};

std::vector<WeightedEdge> collect_undirected_edges(const Graph &G) {
  require_undirected(G, "collect_undirected_edges");
  std::vector<WeightedEdge> edges;
  for (int u = 0; u < G.n; ++u) {
    for (size_t i = 0; i < G.out_adj[u].size(); ++i) {
      int v = G.out_adj[u][i];
      if (u == v)
        continue;
      if (u < v) {
        edges.push_back({u, v, G.weights[u][i]});
      }
    }
  }
  return edges;
}

struct GraphView {
  std::vector<std::unordered_set<int>> out_adj_sets;
  std::vector<std::unordered_set<int>> in_adj_sets;
  std::vector<std::unordered_map<int, double>> out_weights;
  std::vector<std::unordered_map<int, double>> in_weights;
  std::vector<int> out_degree;
  std::vector<int> in_degree;
};

GraphView build_graph_view(const Graph &G) {
  GraphView view;
  view.out_adj_sets.resize(G.n);
  view.in_adj_sets.resize(G.n);
  view.out_weights.resize(G.n);
  view.in_weights.resize(G.n);
  view.out_degree.resize(G.n, 0);
  view.in_degree.resize(G.n, 0);
  for (int u = 0; u < G.n; ++u) {
    view.out_degree[u] = static_cast<int>(G.out_adj[u].size());
    for (size_t i = 0; i < G.out_adj[u].size(); ++i) {
      int v = G.out_adj[u][i];
      double w = G.weights[u][i];
      view.out_adj_sets[u].insert(v);
      view.out_weights[u][v] = w;
      view.in_adj_sets[v].insert(u);
      view.in_weights[v][u] = w;
      view.in_degree[v]++;
    }
  }
  return view;
}

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

std::vector<int> connected_components_union_find(const Graph &G) {
  require_undirected(G, "connected_components_union_find");
  const int n = G.n;
  DisjointSet dsu(n);
  for (int u = 0; u < n; ++u) {
    for (int v : G.out_adj[u]) {
      dsu.unite(u, v);
    }
  }
  std::unordered_map<int, int> comp_map;
  std::vector<int> components(n, -1);
  int next_id = 0;
  for (int i = 0; i < n; ++i) {
    int root = dsu.find(i);
    auto it = comp_map.find(root);
    if (it == comp_map.end()) {
      comp_map[root] = next_id;
      it = comp_map.find(root);
      next_id++;
    }
    components[i] = it->second;
  }
  return components;
}

std::vector<int> connected_components_bfs(const Graph &G) {
  require_undirected(G, "connected_components_bfs");
  const int n = G.n;
  std::vector<int> components(n, -1);
  int comp_id = 0;
  std::queue<int> q;
  for (int i = 0; i < n; ++i) {
    if (components[i] != -1)
      continue;
    components[i] = comp_id;
    q.push(i);
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      for (int v : G.out_adj[u]) {
        if (components[v] == -1) {
          components[v] = comp_id;
          q.push(v);
        }
      }
    }
    comp_id++;
  }
  return components;
}

std::vector<std::tuple<int, int, double>> mst_kruskal(const Graph &G) {
  require_undirected(G, "minimum_spanning_tree");
  auto edges = collect_undirected_edges(G);
  std::sort(edges.begin(), edges.end(),
            [](const WeightedEdge &a, const WeightedEdge &b) {
              if (a.w != b.w)
                return a.w < b.w;
              if (a.u != b.u)
                return a.u < b.u;
              return a.v < b.v;
            });
  DisjointSet dsu(G.n);
  std::vector<std::tuple<int, int, double>> tree;
  tree.reserve(G.n ? G.n - 1 : 0);
  for (const auto &e : edges) {
    if (dsu.find(e.u) != dsu.find(e.v)) {
      dsu.unite(e.u, e.v);
      tree.emplace_back(e.u, e.v, e.w);
    }
  }
  return tree;
}

std::vector<std::tuple<int, int, double>> mst_prim(const Graph &G) {
  require_undirected(G, "minimum_spanning_tree");
  const int n = G.n;
  std::vector<std::tuple<int, int, double>> tree;
  if (n == 0)
    return tree;
  std::vector<bool> in_tree(n, false);
  using PQItem = std::tuple<double, int, int>; // weight, parent, node
  std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;
  for (int start = 0; start < n; ++start) {
    if (in_tree[start])
      continue;
    pq.emplace(0.0, -1, start);
    while (!pq.empty()) {
      auto [weight, parent, node] = pq.top();
      pq.pop();
      if (in_tree[node])
        continue;
      in_tree[node] = true;
      if (parent != -1) {
        tree.emplace_back(parent, node, weight);
      }
      for (size_t i = 0; i < G.out_adj[node].size(); ++i) {
        int nb = G.out_adj[node][i];
        double w = G.weights[node][i];
        if (!in_tree[nb]) {
          pq.emplace(w, node, nb);
        }
      }
    }
  }
  return tree;
}

std::vector<std::tuple<int, int, double>> minimum_spanning_tree(
    const Graph &G, const std::string &algorithm) {
  if (algorithm == "kruskal")
    return mst_kruskal(G);
  if (algorithm == "prim")
    return mst_prim(G);
  throw std::invalid_argument("Unknown MST algorithm: " + algorithm);
}

bool iso_backtrack(int idx, const std::vector<int> &order,
                   std::vector<int> &mapping, std::vector<bool> &used,
                   const GraphView &A, const GraphView &B) {
  const int n = static_cast<int>(order.size());
  if (idx == n)
    return true;
  int u = order[idx];
  for (int v = 0; v < n; ++v) {
    if (used[v])
      continue;
    if (A.out_degree[u] != B.out_degree[v])
      continue;
    if (A.in_degree[u] != B.in_degree[v])
      continue;
    bool ok = true;
    for (int i = 0; i < n; ++i) {
      int mapped = mapping[i];
      if (mapped == -1)
        continue;
      bool out_uw = A.out_adj_sets[u].count(i) > 0;
      bool out_vw = B.out_adj_sets[v].count(mapped) > 0;
      if (out_uw != out_vw) {
        ok = false;
        break;
      }
      if (out_uw) {
        double w1 = A.out_weights[u].at(i);
        double w2 = B.out_weights[v].at(mapped);
        if (std::abs(w1 - w2) > 1e-9) {
          ok = false;
          break;
        }
      }
      bool in_uw = A.in_adj_sets[u].count(i) > 0;
      bool in_vw = B.in_adj_sets[v].count(mapped) > 0;
      if (in_uw != in_vw) {
        ok = false;
        break;
      }
      if (in_uw) {
        double w1 = A.in_weights[u].at(i);
        double w2 = B.in_weights[v].at(mapped);
        if (std::abs(w1 - w2) > 1e-9) {
          ok = false;
          break;
        }
      }
    }
    if (!ok)
      continue;
    mapping[u] = v;
    used[v] = true;
    if (iso_backtrack(idx + 1, order, mapping, used, A, B))
      return true;
    mapping[u] = -1;
    used[v] = false;
  }
  return false;
}

bool graphs_are_isomorphic(const Graph &G1, const Graph &G2) {
  if (G1.n != G2.n)
    return false;
  if (G1.directed != G2.directed)
    return false;
  if (G1.weighted != G2.weighted)
    return false;
  GraphView view1 = build_graph_view(G1);
  GraphView view2 = build_graph_view(G2);
  auto deg_out1 = view1.out_degree;
  auto deg_out2 = view2.out_degree;
  std::sort(deg_out1.begin(), deg_out1.end());
  std::sort(deg_out2.begin(), deg_out2.end());
  if (deg_out1 != deg_out2)
    return false;
  auto deg_in1 = view1.in_degree;
  auto deg_in2 = view2.in_degree;
  std::sort(deg_in1.begin(), deg_in1.end());
  std::sort(deg_in2.begin(), deg_in2.end());
  if (deg_in1 != deg_in2)
    return false;
  const int n = G1.n;
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (view1.out_degree[a] != view1.out_degree[b])
      return view1.out_degree[a] > view1.out_degree[b];
    if (view1.in_degree[a] != view1.in_degree[b])
      return view1.in_degree[a] > view1.in_degree[b];
    return a < b;
  });
  std::vector<int> mapping(n, -1);
  std::vector<bool> used(n, false);
  return iso_backtrack(0, order, mapping, used, view1, view2);
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

  m.def("connected_components_union_find", &connected_components_union_find,
        py::arg("graph"),
        "Connected components via union-find returning component ids per node.");

  m.def("connected_components_bfs", &connected_components_bfs, py::arg("graph"),
        "Connected components via BFS returning component ids per node.");

  m.def("minimum_spanning_tree", &minimum_spanning_tree, py::arg("graph"),
        py::arg("algorithm") = "kruskal",
        "Minimum spanning forest edges using Kruskal or Prim.");

  m.def("graphs_are_isomorphic", &graphs_are_isomorphic, py::arg("graph_a"),
        py::arg("graph_b"),
        "Backtracking-based exact graph isomorphism test.");
}
