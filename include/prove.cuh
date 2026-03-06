#pragma once
#include "./field/alt_bn128.cuh"
#include "./fr-tensor.cuh"
#include "./ioutils.cuh"
#include "ec_operation.cuh"
#include "polynomial.cuh"
#include <vector>
#include <omp.h>

using namespace alt_bn128;

// 结构体声明
struct SumcheckWorkTmp {
    FrTensor tmpw;
    FrTensor a;
    FrTensor b;
    FrTensor c;

    size_t tokens_size;

    SumcheckWorkTmp(size_t tokens_size_);
    
    // 禁止拷贝
    SumcheckWorkTmp(const SumcheckWorkTmp&) = delete;
    SumcheckWorkTmp& operator=(const SumcheckWorkTmp&) = delete;
};

// 内核函数声明
__global__ void sumcheck_kernel(fr_t *X, fr_t *a, fr_t *b, 
                                int in_size, int out_size, fr_t *v);

__global__ void sumcheck_kernel(fr_t *A, fr_t *B, fr_t *a, fr_t *b, fr_t *c, 
                                int in_size, int out_size, fr_t *v);

// 验证函数声明
Fr matmul_sumcheck_phase1(FrTensor &A, FrTensor &B, FrTensor &a, FrTensor &b, FrTensor &c, 
                         fr_t *u, fr_t *v, uint K, uint size, Fr &claim);

void matmul_sumcheck_phase2(FrTensor &A, FrTensor &B, FrTensor &a, FrTensor &b, FrTensor &c, 
                           fr_t *v, uint K, Fr &claim, Fr &eq_accumulate);

Fr eleMul_sumcheck(FrTensor &A, FrTensor &B, FrTensor &a, FrTensor &b, FrTensor &c, 
                  fr_t *u, fr_t *v, uint first_size, uint size, Fr &claim);

// 证明函数声明
std::pair<Fr, Fr> prove_matmul(FrTensor &A, FrTensor &B, 
                  int L, int N, int M, int K, 
                  const std::vector<Fr> u, const std::vector<Fr> v, Fr &claim);

std::pair<Fr, Fr> prove_eleMul(FrTensor &A, FrTensor &B, 
                  int L, int M, int N, 
                  const std::vector<Fr> u, const std::vector<Fr> v, Fr &claim);

Fr combine_claims(FrTensor &X, const vector<Fr> &claims, 
                const vector<vector<Fr>> &u, const vector<Fr> &v, const uint num, const uint size);