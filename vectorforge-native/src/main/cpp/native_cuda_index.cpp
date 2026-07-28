#include "vectorforge_native.hpp"

#ifdef VECTORFORGE_ENABLE_CUDA

#include <cuda.h>
#include <nvrtc.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <chrono>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace vectorforge {

namespace {

using CuInitFn = CUresult(CUDAAPI*)(unsigned int);
using CuDeviceGetCountFn = CUresult(CUDAAPI*)(int*);
using CuDeviceGetFn = CUresult(CUDAAPI*)(CUdevice*, int);
using CuDeviceGetAttributeFn = CUresult(CUDAAPI*)(int*, CUdevice_attribute, CUdevice);
using CuCtxCreateFn = CUresult(CUDAAPI*)(CUcontext*, unsigned int, CUdevice);
using CuCtxDestroyFn = CUresult(CUDAAPI*)(CUcontext);
using CuCtxSetCurrentFn = CUresult(CUDAAPI*)(CUcontext);
using CuMemAllocFn = CUresult(CUDAAPI*)(CUdeviceptr*, std::size_t);
using CuMemFreeFn = CUresult(CUDAAPI*)(CUdeviceptr);
using CuMemcpyHtoDFn = CUresult(CUDAAPI*)(CUdeviceptr, const void*, std::size_t);
using CuMemcpyDtoHFn = CUresult(CUDAAPI*)(void*, CUdeviceptr, std::size_t);
using CuModuleLoadDataExFn = CUresult(CUDAAPI*)(CUmodule*, const void*, unsigned int, CUjit_option*, void**);
using CuModuleGetFunctionFn = CUresult(CUDAAPI*)(CUfunction*, CUmodule, const char*);
using CuModuleUnloadFn = CUresult(CUDAAPI*)(CUmodule);
using CuLaunchKernelFn = CUresult(CUDAAPI*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
using CuEventCreateFn = CUresult(CUDAAPI*)(CUevent*, unsigned int);
using CuEventRecordFn = CUresult(CUDAAPI*)(CUevent, CUstream);
using CuEventSynchronizeFn = CUresult(CUDAAPI*)(CUevent);
using CuEventElapsedTimeFn = CUresult(CUDAAPI*)(float*, CUevent, CUevent);
using CuEventDestroyFn = CUresult(CUDAAPI*)(CUevent);
using CuGetErrorNameFn = CUresult(CUDAAPI*)(CUresult, const char**);
using CuGetErrorStringFn = CUresult(CUDAAPI*)(CUresult, const char**);

#ifdef _WIN32
#define VECTORFORGE_NVRTC_CALL __cdecl
#else
#define VECTORFORGE_NVRTC_CALL
#endif

using NvrtcVersionFn = nvrtcResult(VECTORFORGE_NVRTC_CALL*)(int*, int*);
using NvrtcCreateProgramFn = nvrtcResult(VECTORFORGE_NVRTC_CALL*)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
using NvrtcCompileProgramFn = nvrtcResult(VECTORFORGE_NVRTC_CALL*)(nvrtcProgram, int, const char* const*);
using NvrtcGetProgramLogSizeFn = nvrtcResult(VECTORFORGE_NVRTC_CALL*)(nvrtcProgram, std::size_t*);
using NvrtcGetProgramLogFn = nvrtcResult(VECTORFORGE_NVRTC_CALL*)(nvrtcProgram, char*);
using NvrtcGetPTXSizeFn = nvrtcResult(VECTORFORGE_NVRTC_CALL*)(nvrtcProgram, std::size_t*);
using NvrtcGetPTXFn = nvrtcResult(VECTORFORGE_NVRTC_CALL*)(nvrtcProgram, char*);
using NvrtcDestroyProgramFn = nvrtcResult(VECTORFORGE_NVRTC_CALL*)(nvrtcProgram*);
using NvrtcGetErrorStringFn = const char*(VECTORFORGE_NVRTC_CALL*)(nvrtcResult);

#ifdef _WIN32
using DynamicLibraryHandle = HMODULE;
#else
using DynamicLibraryHandle = void*;
#endif

class CudaException final : public std::runtime_error {
public:
    explicit CudaException(const std::string& message)
            : std::runtime_error(message) {
    }
};

void* resolve_symbol(DynamicLibraryHandle library, const char* symbol_name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(library, symbol_name));
#else
    dlerror();
    return dlsym(library, symbol_name);
#endif
}

DynamicLibraryHandle load_dynamic_library(const char* library_name) {
#ifdef _WIN32
    return LoadLibraryA(library_name);
#else
    return dlopen(library_name, RTLD_NOW | RTLD_LOCAL);
#endif
}

template <typename Fn>
Fn require_symbol(DynamicLibraryHandle library, const char* symbol_name) {
    auto raw = reinterpret_cast<Fn>(resolve_symbol(library, symbol_name));
    if (raw == nullptr) {
        std::string message = std::string("Unable to resolve CUDA symbol: ") + symbol_name;
#ifndef _WIN32
        if (const char* detail = dlerror()) {
            message += std::string(" (") + detail + ")";
        }
#endif
        throw CudaException(message);
    }
    return raw;
}

class CudaDriverApi final {
public:
    static const CudaDriverApi& instance() {
        static const CudaDriverApi api;
        return api;
    }

    std::string format_cuda_error(CUresult result) const {
        const char* name = nullptr;
        const char* description = nullptr;
        cu_get_error_name_(result, &name);
        cu_get_error_string_(result, &description);
        std::ostringstream builder;
        builder << (name != nullptr ? name : "CUDA_ERROR_UNKNOWN");
        if (description != nullptr) {
            builder << ": " << description;
        }
        return builder.str();
    }

    std::string format_nvrtc_error(nvrtcResult result) const {
        const char* description = nvrtc_get_error_string_(result);
        return description != nullptr ? description : "NVRTC error";
    }

    DynamicLibraryHandle cuda_library() const noexcept {
        return cuda_library_;
    }

    DynamicLibraryHandle nvrtc_library() const noexcept {
        return nvrtc_library_;
    }

    CuInitFn cu_init_;
    CuDeviceGetCountFn cu_device_get_count_;
    CuDeviceGetFn cu_device_get_;
    CuDeviceGetAttributeFn cu_device_get_attribute_;
    CuCtxCreateFn cu_ctx_create_;
    CuCtxDestroyFn cu_ctx_destroy_;
    CuCtxSetCurrentFn cu_ctx_set_current_;
    CuMemAllocFn cu_mem_alloc_;
    CuMemFreeFn cu_mem_free_;
    CuMemcpyHtoDFn cu_memcpy_htod_;
    CuMemcpyDtoHFn cu_memcpy_dtoh_;
    CuModuleLoadDataExFn cu_module_load_data_ex_;
    CuModuleGetFunctionFn cu_module_get_function_;
    CuModuleUnloadFn cu_module_unload_;
    CuLaunchKernelFn cu_launch_kernel_;
    CuEventCreateFn cu_event_create_;
    CuEventRecordFn cu_event_record_;
    CuEventSynchronizeFn cu_event_synchronize_;
    CuEventElapsedTimeFn cu_event_elapsed_time_;
    CuEventDestroyFn cu_event_destroy_;
    CuGetErrorNameFn cu_get_error_name_;
    CuGetErrorStringFn cu_get_error_string_;

    NvrtcVersionFn nvrtc_version_;
    NvrtcCreateProgramFn nvrtc_create_program_;
    NvrtcCompileProgramFn nvrtc_compile_program_;
    NvrtcGetProgramLogSizeFn nvrtc_get_program_log_size_;
    NvrtcGetProgramLogFn nvrtc_get_program_log_;
    NvrtcGetPTXSizeFn nvrtc_get_ptx_size_;
    NvrtcGetPTXFn nvrtc_get_ptx_;
    NvrtcDestroyProgramFn nvrtc_destroy_program_;
    NvrtcGetErrorStringFn nvrtc_get_error_string_;

private:
    CudaDriverApi()
            : cuda_library_(load_cuda_driver_library()),
              nvrtc_library_(load_nvrtc_library()) {
        cu_init_ = require_symbol<CuInitFn>(cuda_library_, "cuInit");
        cu_device_get_count_ = require_symbol<CuDeviceGetCountFn>(cuda_library_, "cuDeviceGetCount");
        cu_device_get_ = require_symbol<CuDeviceGetFn>(cuda_library_, "cuDeviceGet");
        cu_device_get_attribute_ = require_symbol<CuDeviceGetAttributeFn>(cuda_library_, "cuDeviceGetAttribute");
        cu_ctx_create_ = require_symbol<CuCtxCreateFn>(cuda_library_, "cuCtxCreate_v2");
        cu_ctx_destroy_ = require_symbol<CuCtxDestroyFn>(cuda_library_, "cuCtxDestroy_v2");
        cu_ctx_set_current_ = require_symbol<CuCtxSetCurrentFn>(cuda_library_, "cuCtxSetCurrent");
        cu_mem_alloc_ = require_symbol<CuMemAllocFn>(cuda_library_, "cuMemAlloc_v2");
        cu_mem_free_ = require_symbol<CuMemFreeFn>(cuda_library_, "cuMemFree_v2");
        cu_memcpy_htod_ = require_symbol<CuMemcpyHtoDFn>(cuda_library_, "cuMemcpyHtoD_v2");
        cu_memcpy_dtoh_ = require_symbol<CuMemcpyDtoHFn>(cuda_library_, "cuMemcpyDtoH_v2");
        cu_module_load_data_ex_ = require_symbol<CuModuleLoadDataExFn>(cuda_library_, "cuModuleLoadDataEx");
        cu_module_get_function_ = require_symbol<CuModuleGetFunctionFn>(cuda_library_, "cuModuleGetFunction");
        cu_module_unload_ = require_symbol<CuModuleUnloadFn>(cuda_library_, "cuModuleUnload");
        cu_launch_kernel_ = require_symbol<CuLaunchKernelFn>(cuda_library_, "cuLaunchKernel");
        cu_event_create_ = require_symbol<CuEventCreateFn>(cuda_library_, "cuEventCreate");
        cu_event_record_ = require_symbol<CuEventRecordFn>(cuda_library_, "cuEventRecord");
        cu_event_synchronize_ = require_symbol<CuEventSynchronizeFn>(cuda_library_, "cuEventSynchronize");
        cu_event_elapsed_time_ = require_symbol<CuEventElapsedTimeFn>(cuda_library_, "cuEventElapsedTime");
        cu_event_destroy_ = require_symbol<CuEventDestroyFn>(cuda_library_, "cuEventDestroy_v2");
        cu_get_error_name_ = require_symbol<CuGetErrorNameFn>(cuda_library_, "cuGetErrorName");
        cu_get_error_string_ = require_symbol<CuGetErrorStringFn>(cuda_library_, "cuGetErrorString");

        nvrtc_version_ = require_symbol<NvrtcVersionFn>(nvrtc_library_, "nvrtcVersion");
        nvrtc_create_program_ = require_symbol<NvrtcCreateProgramFn>(nvrtc_library_, "nvrtcCreateProgram");
        nvrtc_compile_program_ = require_symbol<NvrtcCompileProgramFn>(nvrtc_library_, "nvrtcCompileProgram");
        nvrtc_get_program_log_size_ = require_symbol<NvrtcGetProgramLogSizeFn>(nvrtc_library_, "nvrtcGetProgramLogSize");
        nvrtc_get_program_log_ = require_symbol<NvrtcGetProgramLogFn>(nvrtc_library_, "nvrtcGetProgramLog");
        nvrtc_get_ptx_size_ = require_symbol<NvrtcGetPTXSizeFn>(nvrtc_library_, "nvrtcGetPTXSize");
        nvrtc_get_ptx_ = require_symbol<NvrtcGetPTXFn>(nvrtc_library_, "nvrtcGetPTX");
        nvrtc_destroy_program_ = require_symbol<NvrtcDestroyProgramFn>(nvrtc_library_, "nvrtcDestroyProgram");
        nvrtc_get_error_string_ = require_symbol<NvrtcGetErrorStringFn>(nvrtc_library_, "nvrtcGetErrorString");
    }

    static DynamicLibraryHandle load_cuda_driver_library() {
#ifdef _WIN32
        const char* library_name = "nvcuda.dll";
#else
        const char* library_name = "libcuda.so.1";
#endif
        DynamicLibraryHandle library = load_dynamic_library(library_name);
        if (library == nullptr) {
            std::string message = std::string("Unable to load ") + library_name
                    + ". Install an NVIDIA driver to use the CUDA backend.";
#ifndef _WIN32
            if (const char* detail = dlerror()) {
                message += std::string(" Loader error: ") + detail;
            }
#endif
            throw CudaException(message);
        }
        return library;
    }

    static DynamicLibraryHandle load_nvrtc_library() {
#ifdef _WIN32
        std::vector<std::string> candidates{
                "nvrtc64_120_0.dll",
                "nvrtc64_122_0.dll",
                "nvrtc64_123_0.dll",
                "nvrtc64_124_0.dll"
        };

        if (const char* cuda_path = std::getenv("CUDA_PATH")) {
            std::filesystem::path bin_dir = std::filesystem::path(cuda_path) / "bin";
            if (std::filesystem::exists(bin_dir)) {
                for (const auto& entry : std::filesystem::directory_iterator(bin_dir)) {
                    if (!entry.is_regular_file()) {
                        continue;
                    }
                    const auto file_name = entry.path().filename().string();
                    if (file_name.rfind("nvrtc64_", 0) == 0 && entry.path().extension() == ".dll") {
                        candidates.push_back(entry.path().string());
                    }
                }
            }
        }
#else
        std::vector<std::string> candidates{
                "libnvrtc.so",
                "libnvrtc.so.12"
        };
#endif

        for (const std::string& candidate : candidates) {
            DynamicLibraryHandle library = load_dynamic_library(candidate.c_str());
            if (library != nullptr) {
                return library;
            }
        }

        std::string message = "Unable to load NVRTC. Ensure the CUDA toolkit is installed and on the library path.";
#ifndef _WIN32
        if (const char* detail = dlerror()) {
            message += std::string(" Loader error: ") + detail;
        }
#endif
        throw CudaException(message);
    }

    DynamicLibraryHandle cuda_library_;
    DynamicLibraryHandle nvrtc_library_;
};

struct ConstructionCleanup final {
    const CudaDriverApi& api;
    CUcontext& context;
    CUmodule& module;
    CUevent& kernel_start_event;
    CUevent& kernel_end_event;
    CUdeviceptr& device_vectors;
    CUdeviceptr& device_queries;
    CUdeviceptr& device_scores;
    bool active = true;

    ~ConstructionCleanup() {
        if (!active) {
            return;
        }

        if (context != nullptr) {
            api.cu_ctx_set_current_(context);
        }
        if (device_scores != 0U) {
            api.cu_mem_free_(device_scores);
        }
        if (device_queries != 0U) {
            api.cu_mem_free_(device_queries);
        }
        if (device_vectors != 0U) {
            api.cu_mem_free_(device_vectors);
        }
        if (kernel_start_event != nullptr) {
            api.cu_event_destroy_(kernel_start_event);
        }
        if (kernel_end_event != nullptr) {
            api.cu_event_destroy_(kernel_end_event);
        }
        if (module != nullptr) {
            api.cu_module_unload_(module);
        }
        if (context != nullptr) {
            api.cu_ctx_destroy_(context);
        }
    }

    void release() noexcept {
        active = false;
    }
};

void check_cuda(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) {
        throw CudaException(std::string(operation) + " failed: " + CudaDriverApi::instance().format_cuda_error(result));
    }
}

void check_nvrtc(nvrtcResult result, const char* operation) {
    if (result != NVRTC_SUCCESS) {
        throw CudaException(std::string(operation) + " failed: " + CudaDriverApi::instance().format_nvrtc_error(result));
    }
}

const char* dot_product_kernel_source() {
    return R"(
extern "C" __global__
void vectorforge_dot_product(
        const float* vectors,
        const float* queries,
        float* scores,
        int vector_count,
        int dimensions,
        int query_count
) {
    unsigned int global_index = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int total_scores = static_cast<unsigned int>(vector_count) * static_cast<unsigned int>(query_count);
    if (global_index >= total_scores) {
        return;
    }

    int query_index = static_cast<int>(global_index / static_cast<unsigned int>(vector_count));
    int vector_index = static_cast<int>(global_index % static_cast<unsigned int>(vector_count));
    const float* vector = vectors + (static_cast<unsigned long long>(vector_index) * static_cast<unsigned long long>(dimensions));
    const float* query = queries + (static_cast<unsigned long long>(query_index) * static_cast<unsigned long long>(dimensions));

    float sum = 0.0f;
    for (int d = 0; d < dimensions; ++d) {
        sum += vector[d] * query[d];
    }
    scores[global_index] = sum;
}
)";
}

std::string compile_dot_product_ptx(CUdevice device) {
    const auto& api = CudaDriverApi::instance();

    int major = 0;
    int minor = 0;
    check_cuda(api.cu_device_get_attribute_(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
               "cuDeviceGetAttribute(major)");
    check_cuda(api.cu_device_get_attribute_(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
               "cuDeviceGetAttribute(minor)");

    std::ostringstream architecture;
    architecture << "--gpu-architecture=compute_" << major << minor;
    const std::string architecture_option = architecture.str();

    nvrtcProgram program = nullptr;
    check_nvrtc(api.nvrtc_create_program_(&program, dot_product_kernel_source(), "vectorforge_dot_product.cu", 0, nullptr, nullptr),
                "nvrtcCreateProgram");

    const char* options[] = {
            architecture_option.c_str(),
            "--std=c++17"
    };

    nvrtcResult compile_result = api.nvrtc_compile_program_(program, 2, options);
    if (compile_result != NVRTC_SUCCESS) {
        std::size_t log_size = 0;
        api.nvrtc_get_program_log_size_(program, &log_size);
        std::string log(log_size == 0U ? 1U : log_size, '\0');
        if (log_size > 0U) {
            api.nvrtc_get_program_log_(program, log.data());
        }
        api.nvrtc_destroy_program_(&program);
        throw CudaException("nvrtcCompileProgram failed: " + api.format_nvrtc_error(compile_result) + "\n" + log);
    }

    std::size_t ptx_size = 0;
    check_nvrtc(api.nvrtc_get_ptx_size_(program, &ptx_size), "nvrtcGetPTXSize");
    std::string ptx(ptx_size, '\0');
    check_nvrtc(api.nvrtc_get_ptx_(program, ptx.data()), "nvrtcGetPTX");
    check_nvrtc(api.nvrtc_destroy_program_(&program), "nvrtcDestroyProgram");
    return ptx;
}

std::string cached_dot_product_ptx(CUdevice device) {
    static std::mutex cache_mutex;
    static std::unordered_map<int, std::string> ptx_cache;

    std::lock_guard<std::mutex> lock(cache_mutex);
    auto iterator = ptx_cache.find(device);
    if (iterator != ptx_cache.end()) {
        return iterator->second;
    }

    std::string compiled = compile_dot_product_ptx(device);
    auto [inserted, _] = ptx_cache.emplace(device, compiled);
    return inserted->second;
}

class CudaNativeIndex final : public NativeIndex {
public:
    CudaNativeIndex(std::vector<float> vectors, std::vector<jlong> ids, std::int32_t dimensions)
            : ids_(std::move(ids)),
              dimensions_(dimensions),
              vector_count_(ids_.size()),
              vector_count_int_(static_cast<std::int32_t>(ids_.size())),
              vector_bytes_(required_bytes(vector_count_ * static_cast<std::size_t>(dimensions_), sizeof(float), "device vectors")) {
        if (dimensions_ <= 0) {
            throw std::invalid_argument("dimensions must be positive");
        }
        if (vector_count_ == 0U) {
            throw std::invalid_argument("index data must not be empty");
        }
        if (vectors.size() != vector_count_ * static_cast<std::size_t>(dimensions_)) {
            throw std::invalid_argument("flattened vector buffer does not align with dimensions");
        }

        const auto& api = CudaDriverApi::instance();
        ConstructionCleanup cleanup{api, context_, module_, kernel_start_event_, kernel_end_event_,
                device_vectors_, device_queries_, device_scores_};
        check_cuda(api.cu_init_(0), "cuInit");

        int device_count = 0;
        check_cuda(api.cu_device_get_count_(&device_count), "cuDeviceGetCount");
        if (device_count <= 0) {
            throw std::logic_error("No CUDA device is available for the VectorForge CUDA backend");
        }

        check_cuda(api.cu_device_get_(&device_, 0), "cuDeviceGet");
        check_cuda(api.cu_ctx_create_(&context_, 0U, device_), "cuCtxCreate");
        check_cuda(api.cu_ctx_set_current_(context_), "cuCtxSetCurrent");

        const std::string ptx = cached_dot_product_ptx(device_);
        check_cuda(api.cu_module_load_data_ex_(&module_, ptx.c_str(), 0U, nullptr, nullptr), "cuModuleLoadDataEx");
        check_cuda(api.cu_module_get_function_(&kernel_, module_, "vectorforge_dot_product"), "cuModuleGetFunction");

        check_cuda(api.cu_event_create_(&kernel_start_event_, 0U), "cuEventCreate(kernel_start)");
        check_cuda(api.cu_event_create_(&kernel_end_event_, 0U), "cuEventCreate(kernel_end)");

        check_cuda(api.cu_mem_alloc_(&device_vectors_, static_cast<std::size_t>(vector_bytes_)), "cuMemAlloc(device vectors)");
        check_cuda(api.cu_memcpy_htod_(device_vectors_, vectors.data(), static_cast<std::size_t>(vector_bytes_)),
                   "cuMemcpyHtoD(device vectors)");
        cleanup.release();
    }

    ~CudaNativeIndex() override {
        const auto& api = CudaDriverApi::instance();
        if (context_ != nullptr) {
            api.cu_ctx_set_current_(context_);
        }

        if (device_scores_ != 0U) {
            api.cu_mem_free_(device_scores_);
        }
        if (device_queries_ != 0U) {
            api.cu_mem_free_(device_queries_);
        }
        if (device_vectors_ != 0U) {
            api.cu_mem_free_(device_vectors_);
        }
        if (kernel_start_event_ != nullptr) {
            api.cu_event_destroy_(kernel_start_event_);
        }
        if (kernel_end_event_ != nullptr) {
            api.cu_event_destroy_(kernel_end_event_);
        }
        if (module_ != nullptr) {
            api.cu_module_unload_(module_);
        }
        if (context_ != nullptr) {
            api.cu_ctx_destroy_(context_);
        }
    }

    [[nodiscard]] std::size_t vector_count() const noexcept override {
        return vector_count_;
    }

    [[nodiscard]] std::int32_t dimensions() const noexcept override {
        return dimensions_;
    }

    [[nodiscard]] bool gpu_resident() const noexcept override {
        return true;
    }

    [[nodiscard]] bool supports_metric(Metric metric) const noexcept override {
        return metric == Metric::DotProduct;
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
        if (static_cast<std::size_t>(k) > vector_count_) {
            throw std::invalid_argument("k must be <= vector count");
        }
        if (metric != Metric::DotProduct) {
            throw std::invalid_argument("CUDA backend currently supports only DOT_PRODUCT");
        }

        std::lock_guard<std::mutex> lock(search_mutex_);
        const auto& api = CudaDriverApi::instance();
        check_cuda(api.cu_ctx_set_current_(context_), "cuCtxSetCurrent");

        auto total_start = std::chrono::steady_clock::now();
        ensure_search_capacity(query_count);

        const std::size_t query_bytes = static_cast<std::size_t>(required_bytes(
                static_cast<std::size_t>(query_count) * static_cast<std::size_t>(dimensions_),
                sizeof(float),
                "device queries"
        ));
        const std::size_t score_bytes = static_cast<std::size_t>(required_bytes(
                static_cast<std::size_t>(query_count) * vector_count_,
                sizeof(float),
                "device scores"
        ));

        auto h2d_start = std::chrono::steady_clock::now();
        check_cuda(api.cu_memcpy_htod_(device_queries_, queries, query_bytes), "cuMemcpyHtoD(device queries)");
        auto h2d_end = std::chrono::steady_clock::now();

        void* kernel_arguments[] = {
                &device_vectors_,
                &device_queries_,
                &device_scores_,
                &vector_count_int_,
                &dimensions_,
                &query_count
        };

        const std::size_t total_scores = static_cast<std::size_t>(query_count) * vector_count_;
        if (total_scores > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("query_count * vector_count exceeds the current CUDA kernel limit");
        }

        constexpr unsigned int threads_per_block = 256U;
        const unsigned int blocks = static_cast<unsigned int>((total_scores + threads_per_block - 1U) / threads_per_block);

        check_cuda(api.cu_event_record_(kernel_start_event_, nullptr), "cuEventRecord(kernel_start)");
        check_cuda(api.cu_launch_kernel_(
                           kernel_,
                           blocks,
                           1U,
                           1U,
                           threads_per_block,
                           1U,
                           1U,
                           0U,
                           nullptr,
                           kernel_arguments,
                           nullptr),
                   "cuLaunchKernel");
        check_cuda(api.cu_event_record_(kernel_end_event_, nullptr), "cuEventRecord(kernel_end)");
        check_cuda(api.cu_event_synchronize_(kernel_end_event_), "cuEventSynchronize(kernel_end)");

        float kernel_millis = 0.0f;
        check_cuda(api.cu_event_elapsed_time_(&kernel_millis, kernel_start_event_, kernel_end_event_),
                   "cuEventElapsedTime(kernel)");

        auto d2h_start = std::chrono::steady_clock::now();
        check_cuda(api.cu_memcpy_dtoh_(host_scores_.data(), device_scores_, score_bytes), "cuMemcpyDtoH(device scores)");
        auto d2h_end = std::chrono::steady_clock::now();

        std::vector<SearchCandidate> results = select_top_k_exact(host_scores_.data(), ids_, query_count, k, Metric::DotProduct);
        auto total_end = std::chrono::steady_clock::now();

        if (timings != nullptr) {
            timings->available = true;
            timings->host_to_device_millis =
                    std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count();
            timings->kernel_millis = static_cast<double>(kernel_millis);
            timings->device_to_host_millis =
                    std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count();
            timings->total_millis =
                    std::chrono::duration<double, std::milli>(total_end - total_start).count();
        }

        return results;
    }

private:
    void ensure_search_capacity(std::int32_t query_count) {
        const auto& api = CudaDriverApi::instance();
        if (query_count <= query_capacity_) {
            return;
        }

        if (device_queries_ != 0U) {
            check_cuda(api.cu_mem_free_(device_queries_), "cuMemFree(device queries)");
            device_queries_ = 0U;
        }
        if (device_scores_ != 0U) {
            check_cuda(api.cu_mem_free_(device_scores_), "cuMemFree(device scores)");
            device_scores_ = 0U;
        }

        const std::size_t query_bytes = static_cast<std::size_t>(required_bytes(
                static_cast<std::size_t>(query_count) * static_cast<std::size_t>(dimensions_),
                sizeof(float),
                "device queries"
        ));
        const std::size_t score_bytes = static_cast<std::size_t>(required_bytes(
                static_cast<std::size_t>(query_count) * vector_count_,
                sizeof(float),
                "device scores"
        ));

        check_cuda(api.cu_mem_alloc_(&device_queries_, query_bytes), "cuMemAlloc(device queries)");
        check_cuda(api.cu_mem_alloc_(&device_scores_, score_bytes), "cuMemAlloc(device scores)");
        host_scores_.resize(static_cast<std::size_t>(query_count) * vector_count_);
        query_capacity_ = query_count;
    }

    std::vector<jlong> ids_;
    std::int32_t dimensions_;
    std::size_t vector_count_;
    std::int32_t vector_count_int_ = 0;
    jlong vector_bytes_;

    CUdevice device_{};
    CUcontext context_ = nullptr;
    CUmodule module_ = nullptr;
    CUfunction kernel_ = nullptr;
    CUevent kernel_start_event_ = nullptr;
    CUevent kernel_end_event_ = nullptr;

    CUdeviceptr device_vectors_ = 0U;
    CUdeviceptr device_queries_ = 0U;
    CUdeviceptr device_scores_ = 0U;
    std::int32_t query_capacity_ = 0;
    std::vector<float> host_scores_;
    std::mutex search_mutex_;
};

} // namespace

bool cuda_backend_compiled() noexcept {
    return true;
}

int cuda_device_count() {
    try {
        const auto& api = CudaDriverApi::instance();
        check_cuda(api.cu_init_(0), "cuInit");
        int count = 0;
        check_cuda(api.cu_device_get_count_(&count), "cuDeviceGetCount");
        return count;
    } catch (const std::exception&) {
        return 0;
    }
}

std::shared_ptr<NativeIndex> create_cuda_index(
        std::vector<float> vectors,
        std::vector<jlong> ids,
        std::int32_t dimensions
) {
    auto index = std::make_shared<CudaNativeIndex>(std::move(vectors), std::move(ids), dimensions);
    return index;
}

} // namespace vectorforge

#else

namespace vectorforge {

bool cuda_backend_compiled() noexcept {
    return false;
}

int cuda_device_count() {
    return 0;
}

std::shared_ptr<NativeIndex> create_cuda_index(
        std::vector<float>,
        std::vector<jlong>,
        std::int32_t
) {
    throw std::logic_error("CUDA support was not compiled into the VectorForge native library");
}

} // namespace vectorforge

#endif
