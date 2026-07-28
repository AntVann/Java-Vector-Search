#include "vectorforge_native.hpp"

#ifdef VECTORFORGE_ENABLE_CUVS

#include <cuda_runtime_api.h>
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/brute_force.h>
#include <dlpack/dlpack.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <utility>

namespace vectorforge {
namespace {

void check_cuvs(cuvsError_t status, const char* operation) {
    if (status == CUVS_SUCCESS) {
        return;
    }
    const char* detail = cuvsGetLastErrorText();
    throw std::runtime_error(
            std::string(operation) + " failed" +
            (detail == nullptr ? std::string() : std::string(": ") + detail)
    );
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return;
    }
    throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
}

struct TensorView {
    std::array<std::int64_t, 2> shape;
    std::array<std::int64_t, 2> strides;
    DLManagedTensor tensor{};

    TensorView(
            void* data,
            DLDeviceType device_type,
            std::int64_t rows,
            std::int64_t columns,
            std::uint8_t dtype_code,
            std::uint8_t bits
    ) : shape{rows, columns}, strides{columns, 1} {
        tensor.dl_tensor.data = data;
        tensor.dl_tensor.device = DLDevice{device_type, 0};
        tensor.dl_tensor.ndim = 2;
        tensor.dl_tensor.dtype = DLDataType{dtype_code, bits, 1};
        tensor.dl_tensor.shape = shape.data();
        tensor.dl_tensor.strides = strides.data();
        tensor.dl_tensor.byte_offset = 0;
        tensor.manager_ctx = nullptr;
        tensor.deleter = nullptr;
    }
};

class CuvsResources {
public:
    CuvsResources() {
        check_cuvs(cuvsResourcesCreate(&value_), "cuvsResourcesCreate");
    }

    ~CuvsResources() {
        if (value_ != 0) {
            (void)cuvsResourcesDestroy(value_);
        }
    }

    CuvsResources(const CuvsResources&) = delete;
    CuvsResources& operator=(const CuvsResources&) = delete;

    [[nodiscard]] cuvsResources_t get() const noexcept {
        return value_;
    }

private:
    cuvsResources_t value_ = 0;
};

class CuvsIndexHandle {
public:
    CuvsIndexHandle() {
        check_cuvs(cuvsBruteForceIndexCreate(&value_), "cuvsBruteForceIndexCreate");
    }

    ~CuvsIndexHandle() {
        if (value_ != nullptr) {
            (void)cuvsBruteForceIndexDestroy(value_);
        }
    }

    CuvsIndexHandle(const CuvsIndexHandle&) = delete;
    CuvsIndexHandle& operator=(const CuvsIndexHandle&) = delete;

    [[nodiscard]] cuvsBruteForceIndex_t get() const noexcept {
        return value_;
    }

private:
    cuvsBruteForceIndex_t value_ = nullptr;
};

class CuvsDeviceAllocation {
public:
    CuvsDeviceAllocation(cuvsResources_t resources, std::size_t bytes)
            : resources_(resources), bytes_(bytes) {
        check_cuvs(cuvsRMMAlloc(resources_, &value_, bytes_), "cuvsRMMAlloc");
    }

    ~CuvsDeviceAllocation() {
        if (value_ != nullptr) {
            (void)cuvsRMMFree(resources_, value_, bytes_);
        }
    }

    CuvsDeviceAllocation(const CuvsDeviceAllocation&) = delete;
    CuvsDeviceAllocation& operator=(const CuvsDeviceAllocation&) = delete;

    [[nodiscard]] void* get() const noexcept {
        return value_;
    }

private:
    cuvsResources_t resources_;
    std::size_t bytes_;
    void* value_ = nullptr;
};

cuvsDistanceType to_cuvs_metric(Metric metric) {
    switch (metric) {
        case Metric::Euclidean:
            return L2Expanded;
        case Metric::Cosine:
            return CosineExpanded;
        case Metric::DotProduct:
            return InnerProduct;
    }
    throw std::invalid_argument("unsupported metric");
}

std::size_t metric_index(Metric metric) {
    return static_cast<std::size_t>(metric);
}

class CuvsNativeIndex final : public NativeIndex {
public:
    CuvsNativeIndex(std::vector<float> vectors, std::vector<jlong> ids, std::int32_t dimensions)
            : ids_(std::move(ids)), dimensions_(dimensions) {
        if (dimensions_ <= 0) {
            throw std::invalid_argument("dimensions must be positive");
        }
        if (vectors.empty() || ids_.empty()) {
            throw std::invalid_argument("index data must not be empty");
        }
        if (vectors.size() % static_cast<std::size_t>(dimensions_) != 0U) {
            throw std::invalid_argument("flattened vector buffer does not align with dimensions");
        }
        if (ids_.size() != vectors.size() / static_cast<std::size_t>(dimensions_)) {
            throw std::invalid_argument("ids length must match vector count");
        }

        const std::size_t dataset_bytes = vectors.size() * sizeof(float);
        dataset_ = std::make_unique<CuvsDeviceAllocation>(resources_.get(), dataset_bytes);
        cudaStream_t stream = nullptr;
        check_cuvs(cuvsStreamGet(resources_.get(), &stream), "cuvsStreamGet");
        check_cuda(
                cudaMemcpyAsync(
                        dataset_->get(),
                        vectors.data(),
                        dataset_bytes,
                        cudaMemcpyHostToDevice,
                        stream
                ),
                "cudaMemcpyAsync dataset"
        );

        TensorView dataset(
                dataset_->get(),
                kDLCUDA,
                static_cast<std::int64_t>(ids_.size()),
                dimensions_,
                kDLFloat,
                32
        );
        for (Metric metric : {Metric::Euclidean, Metric::Cosine, Metric::DotProduct}) {
            auto index = std::make_unique<CuvsIndexHandle>();
            check_cuvs(
                    cuvsBruteForceBuild(
                            resources_.get(),
                            &dataset.tensor,
                            to_cuvs_metric(metric),
                            0.0F,
                            index->get()
                    ),
                    "cuvsBruteForceBuild"
            );
            indices_[metric_index(metric)] = std::move(index);
        }
        check_cuvs(cuvsStreamSync(resources_.get()), "cuvsStreamSync");
    }

    [[nodiscard]] std::size_t vector_count() const noexcept override {
        return ids_.size();
    }

    [[nodiscard]] std::int32_t dimensions() const noexcept override {
        return dimensions_;
    }

    [[nodiscard]] bool gpu_resident() const noexcept override {
        return true;
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

        std::lock_guard<std::mutex> lock(search_mutex_);
        const auto started = std::chrono::steady_clock::now();
        const std::size_t query_elements =
                static_cast<std::size_t>(query_count) * static_cast<std::size_t>(dimensions_);
        const std::size_t result_elements =
                static_cast<std::size_t>(query_count) * static_cast<std::size_t>(k);
        const std::size_t query_bytes = query_elements * sizeof(float);
        const std::size_t neighbor_bytes = result_elements * sizeof(std::int64_t);
        const std::size_t distance_bytes = result_elements * sizeof(float);

        CuvsDeviceAllocation device_queries(resources_.get(), query_bytes);
        CuvsDeviceAllocation device_neighbors(resources_.get(), neighbor_bytes);
        CuvsDeviceAllocation device_distances(resources_.get(), distance_bytes);

        cudaStream_t stream = nullptr;
        check_cuvs(cuvsStreamGet(resources_.get(), &stream), "cuvsStreamGet");
        check_cuda(
                cudaMemcpyAsync(device_queries.get(), queries, query_bytes, cudaMemcpyHostToDevice, stream),
                "cudaMemcpyAsync queries"
        );

        TensorView queries_tensor(
                device_queries.get(), kDLCUDA, query_count, dimensions_, kDLFloat, 32
        );
        TensorView neighbors_tensor(
                device_neighbors.get(), kDLCUDA, query_count, k, kDLInt, 64
        );
        TensorView distances_tensor(
                device_distances.get(), kDLCUDA, query_count, k, kDLFloat, 32
        );
        const cuvsFilter no_filter{0, NO_FILTER};
        check_cuvs(
                cuvsBruteForceSearch(
                        resources_.get(),
                        indices_[metric_index(metric)]->get(),
                        &queries_tensor.tensor,
                        &neighbors_tensor.tensor,
                        &distances_tensor.tensor,
                        no_filter
                ),
                "cuvsBruteForceSearch"
        );

        std::vector<std::int64_t> neighbors(result_elements);
        std::vector<float> distances(result_elements);
        check_cuda(
                cudaMemcpyAsync(
                        neighbors.data(),
                        device_neighbors.get(),
                        neighbor_bytes,
                        cudaMemcpyDeviceToHost,
                        stream
                ),
                "cudaMemcpyAsync neighbors"
        );
        check_cuda(
                cudaMemcpyAsync(
                        distances.data(),
                        device_distances.get(),
                        distance_bytes,
                        cudaMemcpyDeviceToHost,
                        stream
                ),
                "cudaMemcpyAsync distances"
        );
        check_cuvs(cuvsStreamSync(resources_.get()), "cuvsStreamSync");

        std::vector<SearchCandidate> results;
        results.reserve(result_elements);
        for (std::size_t i = 0; i < result_elements; ++i) {
            const std::int64_t row = neighbors[i];
            if (row < 0 || static_cast<std::size_t>(row) >= ids_.size()) {
                throw std::runtime_error("cuVS returned an out-of-range neighbor index");
            }
            float score = distances[i];
            if (metric == Metric::Cosine) {
                score = 1.0F - score;
            }
            results.push_back(SearchCandidate{ids_[static_cast<std::size_t>(row)], score});
        }

        if (timings != nullptr) {
            *timings = {};
            timings->total_millis =
                    std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started
                    ).count();
            timings->available = true;
        }
        return results;
    }

private:
    CuvsResources resources_;
    std::unique_ptr<CuvsDeviceAllocation> dataset_;
    std::array<std::unique_ptr<CuvsIndexHandle>, 3> indices_;
    std::vector<jlong> ids_;
    std::int32_t dimensions_;
    std::mutex search_mutex_;
};

} // namespace

bool cuvs_backend_compiled() noexcept {
    return true;
}

std::string cuvs_version() {
    std::uint16_t major = 0;
    std::uint16_t minor = 0;
    std::uint16_t patch = 0;
    check_cuvs(cuvsVersionGet(&major, &minor, &patch), "cuvsVersionGet");
    std::ostringstream value;
    value << major << '.' << minor << '.' << patch;
    return value.str();
}

std::shared_ptr<NativeIndex> create_cuvs_index(
        std::vector<float> vectors,
        std::vector<jlong> ids,
        std::int32_t dimensions
) {
    return std::make_shared<CuvsNativeIndex>(std::move(vectors), std::move(ids), dimensions);
}

} // namespace vectorforge

#else

namespace vectorforge {

bool cuvs_backend_compiled() noexcept {
    return false;
}

std::string cuvs_version() {
    throw std::logic_error("cuVS support was not compiled into the VectorForge native library");
}

std::shared_ptr<NativeIndex> create_cuvs_index(
        std::vector<float>,
        std::vector<jlong>,
        std::int32_t
) {
    throw std::logic_error("cuVS support was not compiled into the VectorForge native library");
}

} // namespace vectorforge

#endif
