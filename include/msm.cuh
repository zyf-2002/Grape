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

__global__ void naive_msm(jacob_t *P, affine_t *out, size_t N, size_t W){

    uint global_id = GET_GLOBAL_ID();
    if(global_id >= N) return;
    jacob_t tmp = P[global_id];
    jacob_t res;
    res.inf();
    
    for(int w = 0; w < W; w++){
        out[w * N + global_id] = res;
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

vector<bn128> precompute_generators(uint N, uint W, affine_t *out){ 
    vector<bn128> generators;
    
    generators.resize(N);
    for(int i = 0; i < N; i++){
        generators[i] = bn128::random_element();
    }
    jacob_t *gpu_points;
    
    cudaMalloc((void **)&gpu_points, sizeof(jacob_t) * N);
    cudaMemcpy(gpu_points, generators.data(), sizeof(jacob_t) * N, cudaMemcpyHostToDevice);
    

    CUDA_TIMER_START(pre_points);
    naive_msm<<<(N + 255) / 256, 256>>>(gpu_points, out, N, W);
    CUDA_DEBUG;
    CUDA_TIMER_STOP(pre_points);

    cudaFree(gpu_points);
    return generators;
}



