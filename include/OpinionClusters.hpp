#ifndef ODSP_OPINIONCLUSTERS_HPP
#define ODSP_OPINIONCLUSTERS_HPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

// Groups of nearly equal continuous opinions -- the natural observable of
// bounded-confidence models, which end in one or several separated clusters.
//
// opinion_clusters() sorts the opinions and starts a new cluster wherever two
// consecutive values differ by more than `gap`. After a bounded-confidence run
// has converged, clusters are internally (almost) equal and separated by more
// than the confidence bound, so any gap between the convergence tolerance and
// the bound gives the same answer; 1e-3 is a safe choice for opinions in [0, 1].
//
// "Major" clusters: models also leave a few stragglers in tiny clusters, so
// results are usually reported as the number of clusters holding at least a
// given fraction of the nodes (count_clusters(..., min_fraction)).

struct OpinionCluster {
    double      position = 0.0;   // mean opinion of the members
    double      min = 0.0, max = 0.0;
    std::size_t size = 0;
};

inline std::vector<OpinionCluster> opinion_clusters(std::vector<double> opinions, double gap) {
    if (!(gap >= 0.0)) throw std::invalid_argument("opinion_clusters: gap must be >= 0.");
    std::vector<OpinionCluster> clusters;
    if (opinions.empty()) return clusters;
    std::sort(opinions.begin(), opinions.end());

    double sum = 0.0;
    OpinionCluster current;
    current.min = opinions.front();
    for (std::size_t i = 0; i < opinions.size(); ++i) {
        if (i > 0 && opinions[i] - opinions[i - 1] > gap) {
            current.max = opinions[i - 1];
            current.position = sum / static_cast<double>(current.size);
            clusters.push_back(current);
            current = OpinionCluster{};
            current.min = opinions[i];
            sum = 0.0;
        }
        sum += opinions[i];
        ++current.size;
    }
    current.max = opinions.back();
    current.position = sum / static_cast<double>(current.size);
    clusters.push_back(current);
    return clusters;
}

// Number of clusters with at least min_fraction of all nodes (all clusters if 0).
inline std::size_t count_clusters(const std::vector<double>& opinions, double gap, double min_fraction = 0.0) {
    const double threshold = min_fraction * static_cast<double>(opinions.size());
    std::size_t count = 0;
    for (const OpinionCluster& c : opinion_clusters(opinions, gap))
        if (static_cast<double>(c.size) >= threshold) ++count;
    return count;
}

// Fraction of the nodes in the largest cluster.
inline double largest_cluster_fraction(const std::vector<double>& opinions, double gap) {
    if (opinions.empty()) return 0.0;
    std::size_t best = 0;
    for (const OpinionCluster& c : opinion_clusters(opinions, gap)) best = std::max(best, c.size);
    return static_cast<double>(best) / static_cast<double>(opinions.size());
}

#endif // ODSP_OPINIONCLUSTERS_HPP
