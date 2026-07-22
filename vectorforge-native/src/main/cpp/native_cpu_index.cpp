#include "vectorforge_native.hpp"

#include <cmath>
#include <cstddef>
#include <utility>

namespace vectorforge {

namespace {

class CpuNativeIndex final : public NativeIndex {
public:
    CpuNativeIndex(std::vector<float> vectors, std::vector<float> norms, std::vector<jlong> ids, std::int32_t dimensions)
            : vectors_(std::move(vectors)),
              norms_(std::move(norms)),
              ids_(std::move(ids)),
              dimensions_(dimensions) {
        if (dimensions_ <= 0) {
            throw std::invalid_argument("dimensions must be positive");
        }
        if (vectors_.empty() || ids_.empty()) {
            throw std::invalid_argument("index data must not be empty");
        }
        if (vectors_.size() % static_cast<std::size_t>(dimensions_) != 0U) {
            throw std::invalid_argument("flattened vector buffer does not align with dimensions");
        }
        if (ids_.size() != vector_count()) {
            throw std::invalid_argument("ids length must match vector count");
        }
        if (norms_.size() != ids_.size()) {
            throw std::invalid_argument("norms length must match vector count");
        }
    }

    [[nodiscard]] std::size_t vector_count() const noexcept override {
        return ids_.size();
    }

    [[nodiscard]] std::int32_t dimensions() const noexcept override {
        return dimensions_;
    }

    [[nodiscard]] bool gpu_resident() const noexcept override {
        return false;
    }

    [[nodiscard]] bool supports_metric(Metric) const noexcept override {
        return true;
    }

    std::vector<SearchCandidate> search(
            const float* queries,
            std::int32_t query_count,
            std::int32_t k,
            Metric metric,
            SearchTimings* timings
    ) override {
        if (queries == nullptr) {
            throw std::invalid_argument("queries buffer must not be null");
        }
        if (query_count <= 0) {
            throw std::invalid_argument("query_count must be positive");
        }
        if (k <= 0) {
            throw std::invalid_argument("k must be positive");
        }
        if (static_cast<std::size_t>(k) > vector_count()) {
            throw std::invalid_argument("k must be <= vector count");
        }

        if (timings != nullptr) {
            *timings = {};
        }

        std::vector<SearchCandidate> all_results;
        all_results.reserve(static_cast<std::size_t>(query_count) * static_cast<std::size_t>(k));

        for (std::int32_t query_index = 0; query_index < query_count; ++query_index) {
            const float* query = queries + (static_cast<std::size_t>(query_index) * static_cast<std::size_t>(dimensions_));
            const float query_norm = metric == Metric::Cosine ? norm(query, dimensions_) : 0.0f;

            std::vector<SearchCandidate> heap;
            heap.reserve(static_cast<std::size_t>(k));

            for (std::size_t vector_index = 0; vector_index < vector_count(); ++vector_index) {
                const std::size_t offset = vector_index * static_cast<std::size_t>(dimensions_);
                const float score = compute_score(query, query_norm, vector_index, offset, metric);
                SearchCandidate candidate{ids_[vector_index], score};

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

private:
    [[nodiscard]] float compute_score(
            const float* query,
            float query_norm,
            std::size_t vector_index,
            std::size_t vector_offset,
            Metric metric
    ) const {
        switch (metric) {
            case Metric::Euclidean:
                return squared_euclidean_distance(vectors_.data() + vector_offset, query, dimensions_);
            case Metric::Cosine:
                return cosine_similarity(
                        vectors_.data() + vector_offset,
                        query,
                        dimensions_,
                        norms_[vector_index],
                        query_norm
                );
            case Metric::DotProduct:
                return dot_product(vectors_.data() + vector_offset, query, dimensions_);
        }
        throw std::invalid_argument("unsupported metric");
    }

    static float squared_euclidean_distance(const float* left, const float* right, std::int32_t dimensions) {
        float sum = 0.0f;
        for (std::int32_t i = 0; i < dimensions; ++i) {
            const float delta = left[i] - right[i];
            sum += delta * delta;
        }
        return sum;
    }

    static float dot_product(const float* left, const float* right, std::int32_t dimensions) {
        float sum = 0.0f;
        for (std::int32_t i = 0; i < dimensions; ++i) {
            sum += left[i] * right[i];
        }
        return sum;
    }

    static float cosine_similarity(
            const float* left,
            const float* right,
            std::int32_t dimensions,
            float left_norm,
            float right_norm
    ) {
        if (left_norm == 0.0f || right_norm == 0.0f) {
            return 0.0f;
        }
        return dot_product(left, right, dimensions) / (left_norm * right_norm);
    }

    static float norm(const float* values, std::int32_t dimensions) {
        float sum = 0.0f;
        for (std::int32_t i = 0; i < dimensions; ++i) {
            sum += values[i] * values[i];
        }
        return std::sqrt(sum);
    }

    std::vector<float> vectors_;
    std::vector<float> norms_;
    std::vector<jlong> ids_;
    std::int32_t dimensions_;
};

std::vector<float> compute_norms(const std::vector<float>& vectors, std::int32_t dimensions, std::size_t vector_count) {
    std::vector<float> norms(vector_count);
    for (std::size_t i = 0; i < vector_count; ++i) {
        const float* vector = vectors.data() + (i * static_cast<std::size_t>(dimensions));
        float sum = 0.0f;
        for (std::int32_t d = 0; d < dimensions; ++d) {
            sum += vector[d] * vector[d];
        }
        norms[i] = std::sqrt(sum);
    }
    return norms;
}
} // namespace

std::shared_ptr<NativeIndex> create_cpu_index(
        std::vector<float> vectors,
        std::vector<jlong> ids,
        std::int32_t dimensions
) {
    auto norms = compute_norms(vectors, dimensions, ids.size());
    return std::make_shared<CpuNativeIndex>(std::move(vectors), std::move(norms), std::move(ids), dimensions);
}

} // namespace vectorforge
