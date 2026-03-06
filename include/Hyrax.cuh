#pragma once
#include <iostream>
#include "ioutils.cuh"
#include "./field/alt_bn128.cuh"
#include "fr-tensor.cuh"
#include "ec_operation.cuh"
#include <vector>
#include <omp.h>

using namespace alt_bn128;
using namespace std;

// 设备函数声明
__device__ __forceinline__ fp_t shfl_down_Fp(
    const fp_t &a,
    unsigned int offset,
    int width,
    unsigned mask);

__device__ __forceinline__ jacob_t shfl_down_G1(
    const jacob_t &p,
    unsigned int offset,
    int width,
    unsigned mask);

__device__ __forceinline__ bucket_t shfl_down_xyzz(
    const bucket_t &p,
    unsigned int offset,
    int width,
    unsigned mask);

// 内核函数声明
__global__ void int_windows_sum(
    int* scalars, affine_t *point, jacob_t *out, uint size, uint N, uint npoints);

__global__ void int_windows_reduce(jacob_t *in, jacob_t *out);

__global__ void single_windows_sum(
    fr_t* scalars, affine_t *point, jacob_t *out, uint N, uint npoints);

__global__ void single_windows_reduce(
    jacob_t *in, jacob_t *out, uint num);

__global__ void many_windows_sum(
    fr_t* scalars, affine_t *point, jacob_t *out, uint npoints);

__global__ void many_windows_reduce(
    jacob_t *in, jacob_t *out, uint num);

__global__ void affine_to_jacob(
    jacob_t *d, affine_t *h, uint size);

// 辅助函数声明
void single_commit(fr_t *d, affine_t *g, jacob_t *com, uint N, uint npoints);

// 类声明
class Hyrax_proof {
public:
    size_t z_size; 
    Fr result;
    Fr *z;
    bn128 commit_d;
    bn128 commit_ad;

    Hyrax_proof(Fr result_, size_t z_len, bn128 commit_d_, bn128 commit_ad_);
    ~Hyrax_proof();
    Hyrax_proof(const Hyrax_proof& other);
    Hyrax_proof& operator=(const Hyrax_proof& other);
};

class Hyrax {
public:
    uint npoints;         // 点大小（没有预计算之前的数量）
    bn128 G;
    affine_t *g_affine;

    Hyrax(uint npoints_, affine_t *g, bn128 &G_);
    ~Hyrax();

    jacob_t* commit(int *tensor, uint size, uint N);
    jacob_t* commit(FrTensor &tensor, uint size, uint N);
    Hyrax_proof open(FrTensor &tensor,const vector<Fr> eval_point_, Fr c, uint size, uint N, uint layer);
};