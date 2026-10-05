#ifndef ODSP_EDGELISTIO_HPP
#define ODSP_EDGELISTIO_HPP

#include <cstddef>
#include <fstream>
#include <istream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "EdgeList.hpp"

// Reading and writing undirected edge lists as text, one edge per line:
//
//   # comment lines start with '#' or '%'
//   alice bob
//   7 12  0.5  1286600000      <- extra columns (weights, timestamps) are ignored
//   3,4                        <- ',' ';' and tabs also separate fields
//
// This covers the common formats of network repositories (SNAP, KONECT,
// NetworkX's write_edgelist, plain CSV without a header). A CSV header line
// such as "source,target" must be removed or commented out -- it would be read
// as an edge between nodes "source" and "target".
//
// Node labels may be any token (numbers or names). They are mapped to indices
// 0..nodes-1 in order of first appearance, and `labels` maps back. Self-loops
// and repeated edges (in either orientation) are dropped and counted, so the
// result is a simple graph ready for AdjacencyGraph. Nodes without edges cannot
// be represented in an edge list.

struct EdgeListFile {
    std::size_t              nodes = 0;
    EdgeList                 edges;    // simple: no self-loops or repeated edges
    std::vector<std::string> labels;   // labels[i] = original label of node i
    std::size_t              self_loops_dropped = 0;
    std::size_t              duplicates_dropped = 0;
};

inline EdgeListFile read_edge_list(std::istream& in, const std::string& source = "input") {
    EdgeListFile out;
    std::unordered_map<std::string, std::size_t> index;
    auto node_id = [&](const std::string& label) {
        const auto it = index.find(label);
        if (it != index.end()) return it->second;
        const std::size_t id = out.labels.size();
        index.emplace(label, id);
        out.labels.push_back(label);
        return id;
    };

    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        for (char& c : line)
            if (c == ',' || c == ';' || c == '\t' || c == '\r') c = ' ';
        std::istringstream fields(line);
        std::string a, b;
        if (!(fields >> a)) continue;               // blank line
        if (a[0] == '#' || a[0] == '%') continue;   // comment
        if (!(fields >> b))
            throw std::runtime_error(source + ":" + std::to_string(line_no) +
                                     ": expected two node labels, got only '" + a + "'.");
        const std::size_t u = node_id(a);   // two statements: fix the id order
        const std::size_t v = node_id(b);
        out.edges.emplace_back(u, v);
    }

    out.nodes = out.labels.size();
    const SimplifyReport report = simplify_edges(out.edges);
    out.self_loops_dropped = report.self_loops;
    out.duplicates_dropped = report.duplicates;
    return out;
}

inline EdgeListFile read_edge_list(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("read_edge_list: cannot open '" + path + "'.");
    return read_edge_list(in, path);
}

// Writes "u v" lines (0-based node indices) after a comment header. Reading the
// file back with read_edge_list gives the same graph, with nodes renumbered in
// order of first appearance (labels[i] is then the original index of node i).
// Useful to save a generated network so other runs or tools can reuse exactly
// the same topology.
inline void write_edge_list(std::ostream& out, std::size_t nodes, const EdgeList& edges) {
    out << "# undirected edge list: " << nodes << " nodes, " << edges.size() << " edges\n";
    for (const Edge& e : edges) out << e.first << ' ' << e.second << '\n';
}

inline void write_edge_list(const std::string& path, std::size_t nodes, const EdgeList& edges) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("write_edge_list: cannot open '" + path + "' for writing.");
    write_edge_list(out, nodes, edges);
}

#endif // ODSP_EDGELISTIO_HPP
