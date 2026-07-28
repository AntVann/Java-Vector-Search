#include "vectorforge_native.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using vectorforge::InvalidHandleException;
using vectorforge::Metric;
using vectorforge::NativeIndex;
using vectorforge::SearchTimings;

namespace {

std::mutex g_index_mutex;
std::unordered_map<jlong, std::shared_ptr<NativeIndex>> g_indices;
jlong g_next_handle = 1L;

void throw_java_exception(JNIEnv* env, const char* class_name, const std::string& message) {
    if (env->ExceptionCheck()) {
        return;
    }
    jclass exception_class = env->FindClass(class_name);
    if (exception_class == nullptr) {
        return;
    }
    env->ThrowNew(exception_class, message.c_str());
}

void translate_exception(JNIEnv* env, const std::exception& ex) {
    if (dynamic_cast<const std::invalid_argument*>(&ex) != nullptr) {
        throw_java_exception(env, "java/lang/IllegalArgumentException", ex.what());
        return;
    }
    if (dynamic_cast<const std::logic_error*>(&ex) != nullptr) {
        throw_java_exception(env, "java/lang/IllegalStateException", ex.what());
        return;
    }
    throw_java_exception(env, "com/vectorforge/nativeindex/NativeInteropException", ex.what());
}

void* require_direct_buffer(JNIEnv* env, jobject buffer, std::string_view name) {
    if (buffer == nullptr) {
        throw std::invalid_argument(std::string(name) + " must not be null");
    }
    void* address = env->GetDirectBufferAddress(buffer);
    if (address == nullptr) {
        throw std::invalid_argument(std::string(name) + " must be a direct buffer");
    }
    return address;
}

jlong require_direct_capacity(JNIEnv* env, jobject buffer, std::string_view name) {
    const jlong capacity = env->GetDirectBufferCapacity(buffer);
    if (capacity < 0) {
        throw std::invalid_argument(std::string(name) + " must be a direct buffer");
    }
    return capacity;
}

Metric require_metric(jint metric_code) {
    switch (metric_code) {
        case 0:
            return Metric::Euclidean;
        case 1:
            return Metric::Cosine;
        case 2:
            return Metric::DotProduct;
        default:
            throw std::invalid_argument("unsupported metric code: " + std::to_string(metric_code));
    }
}

std::shared_ptr<NativeIndex> require_index(jlong handle) {
    if (handle <= 0L) {
        throw InvalidHandleException("native handle must be positive");
    }
    std::lock_guard<std::mutex> lock(g_index_mutex);
    auto iterator = g_indices.find(handle);
    if (iterator == g_indices.end()) {
        throw InvalidHandleException("invalid or closed native handle: " + std::to_string(handle));
    }
    return iterator->second;
}

jlong store_index(std::shared_ptr<NativeIndex> index) {
    std::lock_guard<std::mutex> lock(g_index_mutex);
    if (g_next_handle == std::numeric_limits<jlong>::max()) {
        throw std::runtime_error("native handle space exhausted");
    }
    const jlong handle = g_next_handle++;
    g_indices.emplace(handle, std::move(index));
    return handle;
}

void destroy_index(jlong handle) {
    if (handle <= 0L) {
        throw InvalidHandleException("native handle must be positive");
    }
    std::lock_guard<std::mutex> lock(g_index_mutex);
    if (g_indices.erase(handle) == 0U) {
        throw InvalidHandleException("invalid or already closed native handle: " + std::to_string(handle));
    }
}

std::vector<float> copy_float_buffer(float* values, std::size_t element_count) {
    return std::vector<float>(values, values + element_count);
}

std::vector<jlong> copy_id_buffer(jlong* values, std::size_t element_count) {
    return std::vector<jlong>(values, values + element_count);
}

jlong create_index_common(
        JNIEnv* env,
        jobject vectors_buffer,
        jobject ids_buffer,
        jint vector_count,
        jint dimensions,
        const std::function<std::shared_ptr<NativeIndex>(std::vector<float>, std::vector<jlong>, std::int32_t)>& factory
) {
    if (vector_count <= 0) {
        throw std::invalid_argument("vectorCount must be positive");
    }
    if (dimensions <= 0) {
        throw std::invalid_argument("dimensions must be positive");
    }

    auto* vectors_address = static_cast<float*>(require_direct_buffer(env, vectors_buffer, "vectorsBuffer"));
    auto* ids_address = static_cast<jlong*>(require_direct_buffer(env, ids_buffer, "idsBuffer"));

    const jlong vectors_capacity = require_direct_capacity(env, vectors_buffer, "vectorsBuffer");
    const jlong ids_capacity = require_direct_capacity(env, ids_buffer, "idsBuffer");
    const auto vector_count_size = static_cast<std::size_t>(vector_count);
    const auto dimensions_size = static_cast<std::size_t>(dimensions);
    const jlong required_vector_bytes = vectorforge::required_bytes(vector_count_size * dimensions_size, sizeof(float), "vectorsBuffer");
    const jlong required_id_bytes = vectorforge::required_bytes(vector_count_size, sizeof(jlong), "idsBuffer");

    if (vectors_capacity < required_vector_bytes) {
        throw std::invalid_argument("vectorsBuffer capacity is too small");
    }
    if (ids_capacity < required_id_bytes) {
        throw std::invalid_argument("idsBuffer capacity is too small");
    }

    auto index = factory(
            copy_float_buffer(vectors_address, vector_count_size * dimensions_size),
            copy_id_buffer(ids_address, vector_count_size),
            dimensions
    );
    return store_index(std::move(index));
}

void search_common(
        JNIEnv* env,
        jlong handle,
        jobject queries_buffer,
        jint query_count,
        jint dimensions,
        jint k,
        jint metric_code,
        jobject output_ids_buffer,
        jobject output_scores_buffer,
        jobject timing_buffer,
        bool require_timing
) {
    if (query_count <= 0) {
        throw std::invalid_argument("queryCount must be positive");
    }
    if (dimensions <= 0) {
        throw std::invalid_argument("dimensions must be positive");
    }
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }

    auto index = require_index(handle);
    if (index->dimensions() != dimensions) {
        throw std::invalid_argument("query dimensions do not match native index dimensions");
    }

    auto* queries_address = static_cast<float*>(require_direct_buffer(env, queries_buffer, "queriesBuffer"));
    auto* output_ids_address = static_cast<jlong*>(require_direct_buffer(env, output_ids_buffer, "outputIdsBuffer"));
    auto* output_scores_address = static_cast<float*>(require_direct_buffer(env, output_scores_buffer, "outputScoresBuffer"));

    const jlong queries_capacity = require_direct_capacity(env, queries_buffer, "queriesBuffer");
    const jlong output_ids_capacity = require_direct_capacity(env, output_ids_buffer, "outputIdsBuffer");
    const jlong output_scores_capacity = require_direct_capacity(env, output_scores_buffer, "outputScoresBuffer");

    const auto query_count_size = static_cast<std::size_t>(query_count);
    const auto dimensions_size = static_cast<std::size_t>(dimensions);
    const auto k_size = static_cast<std::size_t>(k);

    const jlong required_query_bytes = vectorforge::required_bytes(query_count_size * dimensions_size, sizeof(float), "queriesBuffer");
    const jlong required_output_id_bytes = vectorforge::required_bytes(query_count_size * k_size, sizeof(jlong), "outputIdsBuffer");
    const jlong required_output_score_bytes = vectorforge::required_bytes(query_count_size * k_size, sizeof(float), "outputScoresBuffer");

    if (queries_capacity < required_query_bytes) {
        throw std::invalid_argument("queriesBuffer capacity is too small");
    }
    if (output_ids_capacity < required_output_id_bytes) {
        throw std::invalid_argument("outputIdsBuffer capacity is too small");
    }
    if (output_scores_capacity < required_output_score_bytes) {
        throw std::invalid_argument("outputScoresBuffer capacity is too small");
    }

    SearchTimings timings;
    SearchTimings* timings_ptr = nullptr;
    double* timing_values = nullptr;
    if (require_timing) {
        timing_values = static_cast<double*>(require_direct_buffer(env, timing_buffer, "timingBuffer"));
        const jlong timing_capacity = require_direct_capacity(env, timing_buffer, "timingBuffer");
        const jlong required_timing_bytes = vectorforge::required_bytes(4U, sizeof(double), "timingBuffer");
        if (timing_capacity < required_timing_bytes) {
            throw std::invalid_argument("timingBuffer capacity is too small");
        }
        timings_ptr = &timings;
    }

    const auto metric = require_metric(metric_code);
    const auto results = index->search(queries_address, query_count, k, metric, timings_ptr);
    if (results.size() != query_count_size * k_size) {
        throw std::runtime_error("native search returned an unexpected result count");
    }

    for (std::size_t i = 0; i < results.size(); ++i) {
        output_ids_address[i] = results[i].id;
        output_scores_address[i] = results[i].score;
    }

    if (timing_values != nullptr) {
        timing_values[0] = timings.host_to_device_millis;
        timing_values[1] = timings.kernel_millis;
        timing_values[2] = timings.device_to_host_millis;
        timing_values[3] = timings.total_millis;
    }
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_vectorforge_nativeindex_NativeBindings_nativeCreateIndex(
        JNIEnv* env,
        jclass,
        jobject vectors_buffer,
        jobject ids_buffer,
        jint vector_count,
        jint dimensions
) {
    try {
        return create_index_common(
                env,
                vectors_buffer,
                ids_buffer,
                vector_count,
                dimensions,
                [](std::vector<float> vectors, std::vector<jlong> ids, std::int32_t dims) {
                    return vectorforge::create_cpu_index(std::move(vectors), std::move(ids), dims);
                }
        );
    } catch (const std::exception& ex) {
        translate_exception(env, ex);
        return 0L;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_vectorforge_nativeindex_NativeBindings_nativeCreateCudaIndex(
        JNIEnv* env,
        jclass,
        jobject vectors_buffer,
        jobject ids_buffer,
        jint vector_count,
        jint dimensions
) {
    try {
        return create_index_common(
                env,
                vectors_buffer,
                ids_buffer,
                vector_count,
                dimensions,
                [](std::vector<float> vectors, std::vector<jlong> ids, std::int32_t dims) {
                    return vectorforge::create_cuda_index(std::move(vectors), std::move(ids), dims);
                }
        );
    } catch (const std::exception& ex) {
        translate_exception(env, ex);
        return 0L;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_vectorforge_nativeindex_NativeBindings_nativeCreateCuvsIndex(
        JNIEnv* env,
        jclass,
        jobject vectors_buffer,
        jobject ids_buffer,
        jint vector_count,
        jint dimensions
) {
    try {
        return create_index_common(
                env,
                vectors_buffer,
                ids_buffer,
                vector_count,
                dimensions,
                [](std::vector<float> vectors, std::vector<jlong> ids, std::int32_t dims) {
                    return vectorforge::create_cuvs_index(std::move(vectors), std::move(ids), dims);
                }
        );
    } catch (const std::exception& ex) {
        translate_exception(env, ex);
        return 0L;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_vectorforge_nativeindex_NativeBindings_nativeSearch(
        JNIEnv* env,
        jclass,
        jlong handle,
        jobject queries_buffer,
        jint query_count,
        jint dimensions,
        jint k,
        jint metric_code,
        jobject output_ids_buffer,
        jobject output_scores_buffer
) {
    try {
        search_common(
                env,
                handle,
                queries_buffer,
                query_count,
                dimensions,
                k,
                metric_code,
                output_ids_buffer,
                output_scores_buffer,
                nullptr,
                false
        );
    } catch (const std::exception& ex) {
        translate_exception(env, ex);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_vectorforge_nativeindex_NativeBindings_nativeSearchCuda(
        JNIEnv* env,
        jclass,
        jlong handle,
        jobject queries_buffer,
        jint query_count,
        jint dimensions,
        jint k,
        jint metric_code,
        jobject output_ids_buffer,
        jobject output_scores_buffer,
        jobject timing_buffer
) {
    try {
        search_common(
                env,
                handle,
                queries_buffer,
                query_count,
                dimensions,
                k,
                metric_code,
                output_ids_buffer,
                output_scores_buffer,
                timing_buffer,
                true
        );
    } catch (const std::exception& ex) {
        translate_exception(env, ex);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_vectorforge_nativeindex_NativeBindings_nativeDestroyIndex(
        JNIEnv* env,
        jclass,
        jlong handle
) {
    try {
        destroy_index(handle);
    } catch (const std::exception& ex) {
        translate_exception(env, ex);
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_vectorforge_nativeindex_NativeBindings_nativeIsCudaCompiled(
        JNIEnv*,
        jclass
) {
    return vectorforge::cuda_backend_compiled() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_vectorforge_nativeindex_NativeBindings_nativeGetCudaDeviceCount(
        JNIEnv* env,
        jclass
) {
    try {
        return static_cast<jint>(vectorforge::cuda_device_count());
    } catch (const std::exception& ex) {
        translate_exception(env, ex);
        return 0;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_vectorforge_nativeindex_NativeBindings_nativeIsCuvsCompiled(
        JNIEnv*,
        jclass
) {
    return vectorforge::cuvs_backend_compiled() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_vectorforge_nativeindex_NativeBindings_nativeGetCuvsVersion(
        JNIEnv* env,
        jclass
) {
    try {
        const std::string version = vectorforge::cuvs_version();
        return env->NewStringUTF(version.c_str());
    } catch (const std::exception& ex) {
        translate_exception(env, ex);
        return nullptr;
    }
}
