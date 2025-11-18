#include <array>
#include <cmath>
#include <cstdint>
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
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  bool unite(int x, int y) {
    int rx = find(x);
    int ry = find(y);
    if (rx == ry)
      return false;
    if (rank[rx] < rank[ry])
      std::swap(rx, ry);
    parent[ry] = rx;
    if (rank[rx] == rank[ry])
      rank[rx]++;
    return true;
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

void radix_sort_edges_by_weight(std::vector<WeightedEdge> &edges,
                                std::vector<int64_t> &keys) {
  const size_t m = edges.size();
  if (m == 0)
    return;
  std::vector<WeightedEdge> scratch_edges(m);
  std::vector<int64_t> scratch_keys(m);
  constexpr int BITS = 8;
  constexpr size_t BUCKETS = 1ull << BITS;
  constexpr uint64_t MASK = BUCKETS - 1;
  for (int shift = 0; shift < 64; shift += BITS) {
    std::array<size_t, BUCKETS> counts{};
    for (size_t i = 0; i < m; ++i) {
      uint64_t key = static_cast<uint64_t>(keys[i]) ^ (1ull << 63);
      const size_t bucket = (key >> shift) & MASK;
      counts[bucket]++;
    }
    size_t total = 0;
    for (size_t b = 0; b < BUCKETS; ++b) {
      size_t c = counts[b];
      counts[b] = total;
      total += c;
    }
    for (size_t i = 0; i < m; ++i) {
      uint64_t key = static_cast<uint64_t>(keys[i]) ^ (1ull << 63);
      const size_t bucket = (key >> shift) & MASK;
      const size_t pos = counts[bucket]++;
      scratch_edges[pos] = edges[i];
      scratch_keys[pos] = keys[i];
    }
    edges.swap(scratch_edges);
    keys.swap(scratch_keys);
  }
}

void sort_weighted_edges(std::vector<WeightedEdge> &edges) {
  constexpr double EPS = 1e-9;
  constexpr size_t RADIX_THRESHOLD = 256;
  std::vector<int64_t> keys;
  keys.reserve(edges.size());
  bool integral = edges.size() >= RADIX_THRESHOLD;
  if (integral) {
    for (const auto &e : edges) {
      double rounded = std::round(e.w);
      if (std::abs(e.w - rounded) > EPS) {
        integral = false;
        break;
      }
      if (rounded < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
          rounded > static_cast<double>(std::numeric_limits<int64_t>::max())) {
        integral = false;
        break;
      }
      keys.push_back(static_cast<int64_t>(rounded));
    }
  }
  if (integral && keys.size() == edges.size()) {
    radix_sort_edges_by_weight(edges, keys);
    return;
  }
  std::sort(edges.begin(), edges.end(),
            [](const WeightedEdge &a, const WeightedEdge &b) {
              if (a.w != b.w)
                return a.w < b.w;
              if (a.u != b.u)
                return a.u < b.u;
              return a.v < b.v;
            });
}

void boruvka_seed_components(const Graph &G,
                             const std::vector<WeightedEdge> &edges,
                             DisjointSet &dsu,
                             std::vector<std::tuple<int, int, double>> &tree,
                             int rounds) {
  if (rounds <= 0)
    return;
  const int n = G.n;
  std::vector<int> best_edge(n, -1);
  std::vector<double> best_weight(n, std::numeric_limits<double>::infinity());
  for (int round = 0; round < rounds; ++round) {
    std::fill(best_edge.begin(), best_edge.end(), -1);
    std::fill(best_weight.begin(), best_weight.end(),
              std::numeric_limits<double>::infinity());
    for (size_t idx = 0; idx < edges.size(); ++idx) {
      const auto &edge = edges[idx];
      int ru = dsu.find(edge.u);
      int rv = dsu.find(edge.v);
      if (ru == rv)
        continue;
      if (edge.w < best_weight[ru]) {
        best_weight[ru] = edge.w;
        best_edge[ru] = static_cast<int>(idx);
      }
      if (edge.w < best_weight[rv]) {
        best_weight[rv] = edge.w;
        best_edge[rv] = static_cast<int>(idx);
      }
    }
    bool any = false;
    for (int u = 0; u < n; ++u) {
      const int idx = best_edge[u];
      if (idx == -1)
        continue;
      const auto &edge = edges[idx];
      if (dsu.unite(edge.u, edge.v)) {
        tree.emplace_back(edge.u, edge.v, edge.w);
        any = true;
      }
    }
    if (!any)
      break;
  }
}

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

std::vector<std::vector<int>> build_in_adj(const Graph &G) {
  std::vector<std::vector<int>> in_adj(G.n);
  for (int u = 0; u < G.n; ++u) {
    for (int v : G.out_adj[u]) {
      in_adj[v].push_back(u);
    }
  }
  return in_adj;
}

std::vector<uint64_t>
weisfeiler_lehman_signatures(const Graph &G,
                             const std::vector<std::vector<int>> &in_adj,
                             int max_iter) {
  const int n = G.n;
  std::vector<uint64_t> colors(n, 0);
  if (n == 0)
    return colors;

  std::vector<uint64_t> next(n, 0);
  for (int u = 0; u < n; ++u) {
    uint64_t base = static_cast<uint64_t>(G.out_adj[u].size());
    base = (base << 32) ^ static_cast<uint64_t>(in_adj[u].size());
    colors[u] = base;
  }
  max_iter = std::max(1, max_iter);
  const uint64_t OUT_MASK = 0x9e3779b97f4a7c15ull;
  const uint64_t IN_MASK = 0xc6a4a7935bd1e995ull;
  for (int iter = 0; iter < max_iter; ++iter) {
    for (int u = 0; u < n; ++u) {
      std::vector<uint64_t> signature;
      signature.reserve(G.out_adj[u].size() + in_adj[u].size());
      for (int v : G.out_adj[u]) {
        signature.push_back(colors[v] ^ OUT_MASK);
      }
      for (int v : in_adj[u]) {
        signature.push_back(colors[v] ^ IN_MASK);
      }
      std::sort(signature.begin(), signature.end());
      uint64_t h = colors[u] ^ 0x51ed270b0b146f7bull;
      for (uint64_t val : signature) {
        h ^= val + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      }
      next[u] = h;
    }
    if (next == colors)
      break;
    colors.swap(next);
  }
  return colors;
}

struct GraphBitsets {
  int words;
  std::vector<uint64_t> out_bits;
  std::vector<uint64_t> in_bits;
};

GraphBitsets build_graph_bitsets(const Graph &G) {
  GraphBitsets bitsets;
  bitsets.words = (G.n == 0) ? 1 : ((G.n + 63) / 64);
  bitsets.out_bits.assign(static_cast<size_t>(G.n) * bitsets.words, 0);
  bitsets.in_bits.assign(static_cast<size_t>(G.n) * bitsets.words, 0);
  for (int u = 0; u < G.n; ++u) {
    for (int v : G.out_adj[u]) {
      size_t out_idx =
          static_cast<size_t>(u) * bitsets.words + static_cast<size_t>(v >> 6);
      bitsets.out_bits[out_idx] |= (1ull << (v & 63));
      size_t in_idx =
          static_cast<size_t>(v) * bitsets.words + static_cast<size_t>(u >> 6);
      bitsets.in_bits[in_idx] |= (1ull << (u & 63));
    }
  }
  return bitsets;
}

inline bool bitset_test(const std::vector<uint64_t> &data, int words, int u,
                        int v) {
  size_t idx =
      static_cast<size_t>(u) * words + static_cast<size_t>(v >> 6);
  const uint64_t mask = 1ull << (v & 63);
  return (data[idx] & mask) != 0;
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
  if (n == 0)
    return {};

  std::vector<int> labels(n);
  std::iota(labels.begin(), labels.end(), 0);
  std::vector<int> buffer = labels;

  bool changed = true;
  int iter = 0;
  const int max_iters = std::max(4, static_cast<int>(std::ceil(std::log2(n + 1))) * 4);
  while (changed && iter < max_iters) {
    changed = false;
    ++iter;
    for (int u = 0; u < n; ++u) {
      int best = labels[u];
      for (int v : G.out_adj[u]) {
        if (labels[v] < best)
          best = labels[v];
      }
      buffer[u] = best;
    }
    for (int u = 0; u < n; ++u) {
      if (buffer[u] != labels[u]) {
        labels[u] = buffer[u];
        changed = true;
      }
    }
    for (int u = 0; u < n; ++u) {
      while (labels[u] != labels[labels[u]]) {
        labels[u] = labels[labels[u]];
      }
    }
  }
  for (int u = 0; u < n; ++u) {
    while (labels[u] != labels[labels[u]]) {
      labels[u] = labels[labels[u]];
    }
  }

  std::unordered_map<int, int> comp_map;
  std::vector<int> components(n, -1);
  int next_id = 0;
  for (int i = 0; i < n; ++i) {
    int root = labels[i];
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

std::vector<std::tuple<int, int, double>> mst_kruskal(const Graph &G) {
  require_undirected(G, "minimum_spanning_tree");
  auto edges = collect_undirected_edges(G);
  DisjointSet dsu(G.n);
  std::vector<std::tuple<int, int, double>> tree;
  tree.reserve(G.n ? G.n - 1 : 0);
  const int rounds = (G.n >= 4096) ? 3 : (G.n >= 512 ? 2 : (G.n >= 128 ? 1 : 0));
  if (rounds > 0)
    boruvka_seed_components(G, edges, dsu, tree, rounds);
  sort_weighted_edges(edges);
  for (const auto &e : edges) {
    if (dsu.unite(e.u, e.v)) {
      tree.emplace_back(e.u, e.v, e.w);
      if (static_cast<int>(tree.size()) == G.n - 1)
        break;
    }
  }
  return tree;
}

class DaryHeap {
public:
  explicit DaryHeap(int n, int degree = 4)
      : d(std::max(2, degree)), position(n, -1),
        keys(n, std::numeric_limits<double>::infinity()) {}

  bool empty() const { return heap.empty(); }

  double key_of(int node) const { return keys[node]; }

  void decrease_key(int node, double value) {
    if (position[node] == -1) {
      heap.push_back(node);
      position[node] = static_cast<int>(heap.size()) - 1;
      keys[node] = value;
      bubble_up(position[node]);
    } else if (value < keys[node]) {
      keys[node] = value;
      bubble_up(position[node]);
    }
  }

  std::pair<int, double> pop_min() {
    int node = heap.front();
    double key = keys[node];
    int last = heap.back();
    heap.pop_back();
    if (!heap.empty()) {
      heap[0] = last;
      position[last] = 0;
      sift_down(0);
    }
    position[node] = -1;
    return {node, key};
  }

  void clear() {
    heap.clear();
    std::fill(position.begin(), position.end(), -1);
    std::fill(keys.begin(), keys.end(),
              std::numeric_limits<double>::infinity());
  }

private:
  int d;
  std::vector<int> heap;
  std::vector<int> position;
  std::vector<double> keys;

  void bubble_up(int idx) {
    while (idx > 0) {
      int parent = (idx - 1) / d;
      if (keys[heap[idx]] < keys[heap[parent]]) {
        std::swap(heap[idx], heap[parent]);
        position[heap[idx]] = idx;
        position[heap[parent]] = parent;
        idx = parent;
      } else {
        break;
      }
    }
  }

  void sift_down(int idx) {
    const int size = static_cast<int>(heap.size());
    while (true) {
      int smallest = idx;
      for (int i = 1; i <= d; ++i) {
        int child = d * idx + i;
        if (child < size &&
            keys[heap[child]] < keys[heap[smallest]]) {
          smallest = child;
        }
      }
      if (smallest == idx)
        break;
      std::swap(heap[idx], heap[smallest]);
      position[heap[idx]] = idx;
      position[heap[smallest]] = smallest;
      idx = smallest;
    }
  }
};

std::vector<std::tuple<int, int, double>> mst_prim(const Graph &G) {
  require_undirected(G, "minimum_spanning_tree");
  const int n = G.n;
  std::vector<std::tuple<int, int, double>> tree;
  if (n == 0)
    return tree;
  std::vector<bool> in_tree(n, false);
  std::vector<int> parent(n, -1);
  DaryHeap heap(n, 8);
  for (int start = 0; start < n; ++start) {
    if (in_tree[start])
      continue;
    heap.decrease_key(start, 0.0);
    parent[start] = -1;
    while (!heap.empty()) {
      auto [node, weight] = heap.pop_min();
      if (in_tree[node])
        continue;
      in_tree[node] = true;
      if (parent[node] != -1) {
        tree.emplace_back(parent[node], node, weight);
      }
      for (size_t i = 0; i < G.out_adj[node].size(); ++i) {
        int nb = G.out_adj[node][i];
        double w = G.weights[node][i];
        if (in_tree[nb])
          continue;
        if (w < heap.key_of(nb)) {
          parent[nb] = node;
          heap.decrease_key(nb, w);
        }
      }
    }
    heap.clear();
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

constexpr double kWeightTolerance = 1e-9;

struct IsoContext {
  const GraphView &A;
  const GraphView &B;
  const GraphBitsets &bits_a;
  const GraphBitsets &bits_b;
  const std::vector<int> &color_a;
  const std::vector<std::vector<int>> &candidates;
  const std::vector<int> &order;
};

bool adjacency_compatible(int u, int v, int depth, const IsoContext &ctx,
                          const std::vector<int> &mapping) {
  for (int i = 0; i < depth; ++i) {
    int other = ctx.order[i];
    int mapped = mapping[other];
    if (mapped == -1)
      continue;
    bool out_a =
        bitset_test(ctx.bits_a.out_bits, ctx.bits_a.words, u, other);
    bool out_b =
        bitset_test(ctx.bits_b.out_bits, ctx.bits_b.words, v, mapped);
    if (out_a != out_b)
      return false;
    if (out_a) {
      double w1 = ctx.A.out_weights[u].at(other);
      double w2 = ctx.B.out_weights[v].at(mapped);
      if (std::abs(w1 - w2) > kWeightTolerance)
        return false;
    }
    bool in_a =
        bitset_test(ctx.bits_a.in_bits, ctx.bits_a.words, u, other);
    bool in_b =
        bitset_test(ctx.bits_b.in_bits, ctx.bits_b.words, v, mapped);
    if (in_a != in_b)
      return false;
    if (in_a) {
      double w1 = ctx.A.in_weights[u].at(other);
      double w2 = ctx.B.in_weights[v].at(mapped);
      if (std::abs(w1 - w2) > kWeightTolerance)
        return false;
    }
  }
  return true;
}

bool iso_backtrack(int depth, const IsoContext &ctx,
                   std::vector<int> &mapping,
                   std::vector<int> &reverse_map) {
  if (depth == static_cast<int>(ctx.order.size()))
    return true;
  int u = ctx.order[depth];
  int color = ctx.color_a[u];
  const auto &bucket = ctx.candidates[color];
  if (bucket.empty())
    return false;
  for (int v : bucket) {
    if (reverse_map[v] != -1)
      continue;
    if (ctx.A.out_degree[u] != ctx.B.out_degree[v])
      continue;
    if (ctx.A.in_degree[u] != ctx.B.in_degree[v])
      continue;
    if (!adjacency_compatible(u, v, depth, ctx, mapping))
      continue;
    mapping[u] = v;
    reverse_map[v] = u;
    if (iso_backtrack(depth + 1, ctx, mapping, reverse_map))
      return true;
    mapping[u] = -1;
    reverse_map[v] = -1;
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
  auto in_adj1 = build_in_adj(G1);
  auto in_adj2 = build_in_adj(G2);
  const int wl_iters =
      std::max(2, static_cast<int>(std::ceil(std::log2(std::max(2, n)))));
  auto wl1 = weisfeiler_lehman_signatures(G1, in_adj1, wl_iters);
  auto wl2 = weisfeiler_lehman_signatures(G2, in_adj2, wl_iters);
  auto sorted1 = wl1;
  auto sorted2 = wl2;
  std::sort(sorted1.begin(), sorted1.end());
  std::sort(sorted2.begin(), sorted2.end());
  if (sorted1 != sorted2)
    return false;
  std::unordered_map<uint64_t, int> color_map;
  int next_color = 0;
  std::vector<int> colors1(n);
  std::vector<int> colors2(n);
  for (int i = 0; i < n; ++i) {
    auto it = color_map.find(wl1[i]);
    if (it == color_map.end()) {
      color_map[wl1[i]] = next_color;
      colors1[i] = next_color;
      next_color++;
    } else {
      colors1[i] = it->second;
    }
  }
  for (int i = 0; i < n; ++i) {
    auto it = color_map.find(wl2[i]);
    if (it == color_map.end()) {
      color_map[wl2[i]] = next_color;
      colors2[i] = next_color;
      next_color++;
    } else {
      colors2[i] = it->second;
    }
  }
  std::vector<int> color_counts(next_color, 0);
  for (int c : colors1)
    color_counts[c]++;
  std::vector<std::vector<int>> candidates(next_color);
  for (int v = 0; v < n; ++v) {
    candidates[colors2[v]].push_back(v);
  }
  for (int color = 0; color < next_color; ++color) {
    if (color_counts[color] != static_cast<int>(candidates[color].size()))
      return false;
    std::sort(candidates[color].begin(), candidates[color].end(),
              [&](int lhs, int rhs) {
                if (view2.out_degree[lhs] != view2.out_degree[rhs])
                  return view2.out_degree[lhs] > view2.out_degree[rhs];
                if (view2.in_degree[lhs] != view2.in_degree[rhs])
                  return view2.in_degree[lhs] > view2.in_degree[rhs];
                return lhs < rhs;
              });
  }
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (color_counts[colors1[a]] != color_counts[colors1[b]])
      return color_counts[colors1[a]] < color_counts[colors1[b]];
    const int deg_a = view1.out_degree[a] + view1.in_degree[a];
    const int deg_b = view1.out_degree[b] + view1.in_degree[b];
    if (deg_a != deg_b)
      return deg_a > deg_b;
    return a < b;
  });
  GraphBitsets bits1 = build_graph_bitsets(G1);
  GraphBitsets bits2 = build_graph_bitsets(G2);
  IsoContext ctx{view1, view2, bits1, bits2, colors1, candidates, order};
  std::vector<int> mapping(n, -1);
  std::vector<int> reverse_map(n, -1);
  return iso_backtrack(0, ctx, mapping, reverse_map);
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