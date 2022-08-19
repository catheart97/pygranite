#pragma once

#include "Config.hpp"

#include <cstdint>
#include <iostream>
#include <mutex>

#ifndef __CUDACC__
#include <chrono>
#endif

/**
 * @brief This file contains useful macros and functions for logging and timing of applications.
 *        Use Config.h to specify options.
 */

// timing
#ifndef __CUDACC__
#define TIMERSTART(label)                                                                          \
    std::chrono::time_point<std::chrono::system_clock> a##label, b##label;                         \
    a##label = std::chrono::system_clock::now();
#else
#define TIMERSTART(label)                                                                          \
    cudaEvent_t start##label, stop##label;                                                         \
    float time##label;                                                                             \
    cudaEventCreate(&start##label);                                                                \
    cudaEventCreate(&stop##label);                                                                 \
    cudaEventRecord(start##label, 0);
#endif

#ifndef __CUDACC__
#define TIMERSTOP(label)                                                                           \
    b##label = std::chrono::system_clock::now();                                                   \
    std::chrono::duration<double> delta##label = b##label - a##label;                              \
    std::cout << "LOG < Timing (" << #label << "): " << delta##label.count() << "s >\n";
#else
#define TIMERSTOP(label)                                                                           \
    cudaEventRecord(stop##label, 0);                                                               \
    cudaEventSynchronize(stop##label);                                                             \
    cudaEventElapsedTime(&time##label, start##label, stop##label);                                 \
    std::cout << "LOG < Timing (" << #label << "): " << time##label << "ms >\n"
#endif

// error handling and shortcuts
#ifdef __NVCC__
#define MY_CUDA_ERROR_CHECK                                                                        \
    {                                                                                              \
        cudaError_t err;                                                                           \
        if ((err = cudaGetLastError()) != cudaSuccess)                                             \
        {                                                                                          \
            std::cerr << "ERROR < CUDA error | " << cudaGetErrorString(err) << " : " << __FILE__   \
                      << ", line " << __LINE__ << ">" << std::endl;                                \
            exit(1);                                                                               \
        }                                                                                          \
    }
#else
#define MY_CUDA_ERROR_CHECK
#endif

namespace my::util
{

#ifdef __NVCC__
__device__ __host__ __forceinline__
#else
inline
#endif
    constexpr char *
    cstrrchr(const char * str, const char c)
{
    char * s{nullptr};
    char * i{const_cast<char *>(str)};
    while (*i != '\0')
    {
        if (*i == c) s = i;
        i++;
    }
    return s;
}
} // namespace my::util

#ifdef _WIN32
#define __FILENAME__                                                                               \
    (my::util::cstrrchr(__FILE__, '\\') ? my::util::cstrrchr(__FILE__, '\\') + 1 : __FILE__)
#else
#define __FILENAME__                                                                               \
    (my::util::cstrrchr(__FILE__, '/') ? my::util::cstrrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#ifdef MY_VLOGGING
#define MY_LOGGING
#define MY_VLOG(x)                                                                                 \
    {                                                                                              \
        std::cout << "LOG < " << __FILENAME__ << " " << __LINE__ << " | " << x << " >\n"           \
                  << std::flush;                                                                   \
    }
#define MY_VLOGG(x, ...)                                                                           \
    printf("LOG < GPU %s %i | "##x " >\n", __FILENAME__, __LINE__, __VA_ARGS__);
#define MY_VLOGE(x)                                                                                \
    {                                                                                              \
        x                                                                                          \
    }
#else
#define MY_VLOG(x)
#define MY_VLOGG(x)
#define MY_VLOGE(x)
#endif

#ifdef MY_LOGGING
#define MY_LOG(x)                                                                                  \
    {                                                                                              \
        std::cout << "LOG < " << __FILENAME__ << " " << __LINE__ << " | " << x << " >\n"           \
                  << std::flush;                                                                   \
    }
#define MY_LOGG(x, ...) printf("LOG < GPU %s %i | "##x " >\n", __FILENAME__, __LINE__, __VA_ARGS__);
#define MY_LOGE(x)                                                                                 \
    {                                                                                              \
        x                                                                                          \
    }
inline std::mutex __my_log;
#define MY_SLOG(x)                                                                                 \
    {                                                                                              \
        std::lock_guard<std::mutex> g(__my_log);                                                   \
        std::cout << "LOG < SYNCED " << __FILENAME__ << " " << __LINE__ << " | " << x << " >"      \
                  << std::endl;                                                                    \
    }
#else
#define MY_LOG(x)
#define MY_LOGG(x)
#define MY_LOGE(x)
#define MY_SLOG(x)
#endif

#define MY_VAR(x) " " #x ": " << x << " "

#define MY_USER_WARNING(x)                                                                         \
    std::cout << "WARNING < USER INPUT | " << x << " | " << __LINE__ << " >" << std::endl;
#define MY_USER_ERROR(x)                                                                           \
    {                                                                                              \
        std::stringstream ss;                                                                      \
        ss << "ERROR < USER INPUT | " << x << " | " << __LINE__ << " >";                           \
        throw std::runtime_error(ss.str());                                                        \
    }
#define MY_RUNTIME_ERROR(x)                                                                        \
    {                                                                                              \
        std::stringstream ss;                                                                      \
        ss << "ERROR < RUNTIME | " << x << " | " << __LINE__ << " >";                              \
        throw std::runtime_error(ss.str());                                                        \
    }

#ifdef __CUDACC__

#define WARP_SIZE 32

template <typename value_t, typename index_t>
__device__ __forceinline__ value_t warp_shfl_down(value_t var, index_t delta)
{
    return __shfl_down_sync(0xFFFFFFFF, var, delta, WARP_SIZE);
}

#endif

#define COMBINE(A, B) A##B