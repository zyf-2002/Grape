#pragma once
#include "./field/alt_bn128.cuh"
#include "libff/common/profiling.hpp"
#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"
#include "libff/algebra/scalar_multiplication/multiexp.hpp"
#include "host_affine.cuh"
#include "ioutils.cuh"

using namespace std;

typedef default_r1cs_ppzksnark_pp ppT;
typedef libff::G1<ppT> bn128;
typedef ppT::Fp_type Fr;

using namespace alt_bn128;

__global__ void naive_msm(jacob_t *out, jacob_t *P, affine_t *Aout, size_t N, size_t W){

    uint global_id = GET_GLOBAL_ID();
    if(global_id >= N) return;
    jacob_t tmp = P[global_id];
    jacob_t res;
    res.inf();
    
    for(int w = 0; w < W; w++){
        out[w * N + global_id] = res;
        Aout[w * N + global_id] = res;
        res.add(tmp);
    }
    
}

template <typename T>
T host_partial_me(T *points, uint inputsize, vector<Fr>::const_iterator begin, vector<Fr>::const_iterator end){
    if(begin >= end) return points[0];
    uint outputsize = (inputsize + 1) / 2;
    T *new_points = new T[outputsize];
    #pragma omp parallel for
    for(int i = 0; i < outputsize; i++){
        uint id0 = i * 2;
        uint id1 = i * 2 + 1;
        if(id1 < inputsize) new_points[i] = points[id0] + *begin * (points[id1] - points[id0]);
        else new_points[i] = points[id0] - *begin * points[id0];
    }
    return host_partial_me(new_points, outputsize, begin + 1, end);    
}

vector<bn128> precompute_generators(uint N, uint W, affine_t *Aout){ 
    vector<bn128> generators;
    //vector<G1_affine> g1_affine;

    generators.resize(N);
    for(int i = 0; i < N; i++){
        generators[i] = bn128::random_element();
    }
    jacob_t *gpu_points;
    jacob_t *out;
    
    cudaMalloc((void **)&gpu_points, sizeof(jacob_t) * N);
    cudaMalloc((void **)&out, sizeof(jacob_t) * N * W);
    cudaMemcpy(gpu_points, generators.data(), sizeof(jacob_t) * N, cudaMemcpyHostToDevice);
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始事件
    cudaEventRecord(start, 0);
    naive_msm<<<(N + 255) / 256, 256>>>(out, gpu_points, Aout, N, W);
    cudaEventRecord(stop, 0);

    // 等待 kernel 完成
    cudaEventSynchronize(stop);

    CUDA_DEBUG;

    // 计算耗时
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    printf("naive_msm kernel time: %f ms\n", elapsed_ms);

    bn128 *gpu_pre = new bn128[N * W];
    bn128 *cpu_pre = new bn128[N * W];
    cudaMemcpy(gpu_pre, out, sizeof(jacob_t) * N * W, cudaMemcpyDeviceToHost);
    

    #pragma omp parallel for
    for(int i=0; i < N; i++){
        bn128 now = bn128::zero();
        for(int j=0; j < W; j++){
            cpu_pre[j * N + i] = now;
            now = now.add(generators[i]);
        }
    }

    for(int i=0; i < N * W; i++){
        assert(cpu_pre[i] == gpu_pre[i]);
    }
    printf("precompute_generators passed\n");
    cudaFree(gpu_points);
    cudaFree(out);
    return generators;
}

// __global__ void sum_g1_points(affine_t *points, int n, jacob_t *gpu_data) {
//    gpu_data->inf();
//    if(gpu_data->is_inf()) printf("inf\n");
//    for(int i=0; i < n; i++) gpu_data->add(points[i]);
// }


