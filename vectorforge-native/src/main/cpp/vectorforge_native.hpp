#pragma once

#include <jni.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace vectorforge {

enum class Metric : std::int32_t {
    Euclidean = 0,
    Cosine = 1,
    DotProduct = 2
};

struct SearchCandidate {
    jlong id;
    float score;
};

struct SearchTimings {
    double host_to_device_millis = 0.0;
    double kernel_millis = 0.0;
    double device_to_host_millis = 0.0;
    double total_millis = 0.0;
    bool available = false;
};

class InvalidHandleException final : public std::logic_error {
public:
    explicit InvalidHandleException(const std::string& message)
            : std::logic_error(message) {
    }
};

class NativeIndex {
public:
    virtual ~NativeIndex() = default;

    [[nodiscard]] virtual std::size_t vector_count() const noexcept = 0;
    [[nodiscard]] virtual std::int32_t dimensions() const noexcept = 0;
    [[nodiscard]] virtual bool gpu_resident() const noexcept = 0;
    [[nodiscard]] virtual bool supports_metric(Metric metric) const noexcept = 0;

    virtual std::vector<SearchCandidate> search(
            const float* queries,
            std::int32_t query_count,
            std::int32_t k,
            Metric metric,
            SearchTimings* timings
    ) = 0;
};

inline bool is_better(const SearchCandidate& candidate, const SearchCandidate& current, Metric metric) {
    if (candidate.score == current.score) {
        return candidate.id < current.id;
    }
    return metric == Metric::Euclidean
            ? candidate.score < current.score
            : candidate.score > current.score;
}
inline bool is_worse(const SearchCandidate& left, const SearchCandidate& right, Metric metric) {
    if (left.score == right.score) {
        return left.id > right.id;
    }
    return metric == Metric::Euclidean
            ? left.score > right.score
            : left.score < right.score;
}

inline void sift_up(std::vector<SearchCandidate>& heap, std::size_t index, Metric metric) {
    std::size_t current = index;
    while (current > 0U) {
        const std::size_t parent = (current - 1U) / 2U;
        if (!is_worse(heap[current], heap[parent], metric)) {
            return;
        }
        std::swap(heap[current], heap[parent]);
        current = parent;
    }
}

inline void sift_down(std::vector<SearchCandidate>& heap, std::size_t index, Metric metric) {
    std::size_t current = index;
    while (true) {
        const std::size_t left = (current * 2U) + 1U;
        if (left >= heap.size()) {
            return;
        }
        const std::size_t right = left + 1U;
        std::size_t worst_child = left;
        if (right < heap.size() && is_worse(heap[right], heap[left], metric)) {
            worst_child = right;
        }
        if (!is_worse(heap[worst_child], heap[current], metric)) {
            return;
        }
        std::swap(heap[current], heap[worst_child]);
        current = worst_child;
    }
}

inline void sort_best_first(std::vector<SearchCandidate>& results, Metric metric) {
    for (std::size_t i = 1U; i < results.size(); ++i) {
        SearchCandidate current = results[i];
        std::size_t j = i;
        while (j > 0U && is_better(current, results[j - 1U], metric)) {
            results[j] = results[j - 1U];
            --j;
        }
        results[j] = current;
    }
}

inline std::vector<SearchCandidate> select_top_k_exact(
        const float* scores,
        const std::vector<jlong>& ids,
        std::int32_t query_count,
        std::int32_t k,
        Metric metric
) {
    if (scores == nullptr) {
        throw std::invalid_argument("scores must not be null");
    }
    if (query_count <= 0) {
        throw std::invalid_argument("query_count must be positive");
    }
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }
    if (ids.empty()) {
        throw std::invalid_argument("ids must not be empty");
    }
    if (static_cast<std::size_t>(k) > ids.size()) {
        throw std::invalid_argument("k must be <= vector count");
    }

    std::vector<SearchCandidate> all_results;
    all_results.reserve(static_cast<std::size_t>(query_count) * static_cast<std::size_t>(k));

    for (std::int32_t query_index = 0; query_index < query_count; ++query_index) {
        std::vector<SearchCandidate> heap;
        heap.reserve(static_cast<std::size_t>(k));
        const float* query_scores = scores + (static_cast<std::size_t>(query_index) * ids.size());

        for (std::size_t vector_index = 0; vector_index < ids.size(); ++vector_index) {
            SearchCandidate candidate{ids[vector_index], query_scores[vector_index]};
            if (heap.size() < static_cast<std::size_t>(k)) {
                heap.push_back(candidate);
                sift_up(heap, heap.size() - 1U, metric);
                continue;
            }

            if (is_better(candidate, heap.front(), metric)) {
                heap.front() = candidate;
                sift_down(heap, 0U, metric);
            }
        }

        sort_best_first(heap, metric);
        all_results.insert(all_results.end(), heap.begin(), heap.end());
    }

    return all_results;
}

inline jlong required_bytes(std::size_t count, std::size_t element_size, std::string_view description) {
    if (count > 0U && element_size > std::numeric_limits<jlong>::max() / static_cast<jlong>(count)) {
        throw std::invalid_argument(std::string(description) + " size overflow");
    }
    return static_cast<jlong>(count * element_size);
}

std::shared_ptr<NativeIndex> create_cpu_index(
        std::vector<float> vectors,
        std::vector<jlong> ids,
        std::int32_t dimensions
);

bool cuda_backend_compiled() noexcept;

int cuda_device_count();

std::shared_ptr<NativeIndex> create_cuda_index(
        std::vector<float> vectors,
        std::vector<jlong> ids,
        std::int32_t dimensions
);

} // namespace vectorforge
