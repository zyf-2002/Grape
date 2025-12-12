#pragma once
#include <iostream>
#include <string>

using namespace std;

#define GET_GLOBAL_ID() blockIdx.x * blockDim.x + threadIdx.x

#define GET_GLOBAL_WARP_ID() (blockIdx.x * blockDim.x + threadIdx.x) / 32
#define GET_WARP_ID() threadIdx.x / 32
#define GET_LANE_ID() (blockIdx.x * blockDim.x + threadIdx.x) % 32
#define GET_TOTAL_THREADS() blockDim.x * gridDim.x

#define CUDA_DEBUG cudaDeviceSynchronize(); std::cout << __LINE__ << " " << cudaGetErrorString(cudaGetLastError()) << std::endl;

#define CUDA_TIMER_START(timer_name)            \
    cudaEvent_t timer_name##_start, timer_name##_stop; \
    cudaEventCreate(&timer_name##_start);      \
    cudaEventCreate(&timer_name##_stop);       \
    cudaEventRecord(timer_name##_start, 0);

#define CUDA_TIMER_STOP(timer_name)                             \
    cudaEventRecord(timer_name##_stop, 0);                     \
    cudaEventSynchronize(timer_name##_stop);                   \
    float timer_name##_elapsed_ms = 0.0f;                      \
    cudaEventElapsedTime(&timer_name##_elapsed_ms, timer_name##_start, timer_name##_stop); \
    printf("%s elapsed: %f ms\n", #timer_name, timer_name##_elapsed_ms); \
    cudaEventDestroy(timer_name##_start);                       \
    cudaEventDestroy(timer_name##_stop);
    

#define CPU_TIMER_START(timer_name)                                        \
    auto timer_name##_start = std::chrono::high_resolution_clock::now();

#define CPU_TIMER_STOP(timer_name)                                         \
    do {                                                                    \
        auto timer_name##_end = std::chrono::high_resolution_clock::now();  \
        double timer_name##_elapsed_ms =                                    \
            std::chrono::duration<double, std::milli>(                      \
                timer_name##_end - timer_name##_start                       \
            ).count();                                                      \
        printf("%s elapsed: %f ms\n", #timer_name, timer_name##_elapsed_ms);\
    } while (0)


void savebin(const string& filename, const void* gpudata, uint size);

ulong findsize(const string& filename);

void loadbin(const string& filename, void* gpudata, ulong size);

uint Log2(uint num);

uint load_data(const string& filename, int **out_ptr);