#pragma once
#include "./field/alt_bn128.cuh"
#include "./ioutils.cuh"
#include "ec_operation.cuh"
#include <vector>
#include <cassert>
#include <stdexcept>
#include <algorithm>

using namespace alt_bn128;

// 内核函数声明
__device__ __forceinline__ fr_t shfl_down_Fr(const fr_t &a, unsigned int offset, int width, unsigned mask);

__global__ void pad_int_kernel(const int *input, int *output, uint N, uint pad_N, uint size);
__global__ void pad_fr_kernel(fr_t *input, fr_t *output, uint N, uint pad_N, uint size);
__global__ void int_to_fr(int *input, fr_t *out, size_t N);
__global__ void uint_to_fr(uint *input, fr_t *out, size_t N);
__global__ void Memory_alignment(fr_t *t, uint gap);
__global__ void Fr_partial_eval_kernel(fr_t *t, fr_t *x, uint dim, uint other_dims, uint in_cur_dim, uint out_cur_dim, uint id, uint window_size, uint gap);
__global__ void Fr_elementwise_add(fr_t* arr1, fr_t* arr2, fr_t* arr_out, uint n);
__global__ void Fr_elementwise_sub(fr_t* arr1, fr_t* arr2, fr_t* arr_out, uint n);
__global__ void Fr_elementwise_mul(fr_t* arr1, fr_t* arr2, fr_t* arr_out, uint n);
__global__ void Fr_broadcast_mul(fr_t* arr, fr_t *x, fr_t* arr_out, uint n);
__global__ void from_kernel(fr_t *arr, uint size);
__global__ void to_kernel(fr_t *arr, uint size);
__global__ void inverse_kernel(fr_t *arr, uint size);
__global__ void Fr_sum_reduction(fr_t *arr, fr_t *output, uint n);

__global__ void sum_by_dim_kernel(fr_t* in, fr_t* out, uint N, uint dim);




// FrTensor 类声明
class FrTensor
{   
public:
    fr_t* gpu_data;
    uint size;
    
    FrTensor(uint size);
    FrTensor(const FrTensor& t);
    FrTensor(uint size, int* cpu_data);
    ~FrTensor();
    
    FrTensor& operator=(const FrTensor& x);
    
    void set_size(uint sz);
    void get_data(uint sz, int *cpu_data);
    void get_data(uint sz, uint *cpu_data);
    void partial_eval(uint N, fr_t *x, uint c, uint id, uint gap);
    Fr operator()(const std::vector<Fr>& u);
    void add_with_size(uint sz, const FrTensor& t);
    void mul_with_size(uint sz, const FrTensor& t);
    void mul_with_size(uint sz, fr_t *x);
    void mul_with_size(uint sz, Fr &x);
    void inverse();
    void from();
    void to();
    Fr sum(uint sz);
    FrTensor sum(uint sz, uint N);  //N是需要sum的那个维度的长度
    Fr operator()(uint idx) const;
    void pad(FrTensor &tmp, uint N, uint pad_N, uint pad_size, const Fr &pad_val);

    FrTensor& operator+=(const FrTensor& t);
    FrTensor& operator+=(const int x);
    FrTensor& operator-=(const FrTensor& t);
    FrTensor& operator*=(const FrTensor& t);
    FrTensor& operator*=(fr_t *x);
    FrTensor& operator*=(const uint x);
    FrTensor& operator*=(Fr &x);
};

// 辅助函数声明
void Fr_partial_me(uint N, FrTensor& t, fr_t *x, uint id, uint cur_dim, uint window_size);
void pad_int(int **int_data, uint N, uint pad_N, uint pad_size);
