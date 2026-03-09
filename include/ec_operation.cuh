#pragma once
#include <vector>
#include "./field/alt_bn128.cuh"
//#include "libff/common/profiling.hpp"
#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"
//#include "libff/algebra/scalar_multiplication/multiexp.hpp"
#include <omp.h>
#include "ioutils.cuh"

using namespace std;
using namespace libsnark;
typedef default_r1cs_ppzksnark_pp ppT;
typedef libff::G1<ppT> bn128;
typedef ppT::Fp_type Fr;

using namespace alt_bn128;

// ---------------- kernel declarations ----------------
__global__ void check_G_equal(const jacob_t *a, const jacob_t *b, int n);
__global__ void naive_msm(jacob_t *P, affine_t *out, size_t N, size_t W);

// ---------------- host function declarations ----------------
vector<bn128> precompute_generators(uint N, uint W, affine_t *out);

vector<Fr> random_vec(uint len);
void generate_random_eval_points(size_t data_size, vector<Fr>& eval_point);

void ReassembleVectors(std::vector<Fr>& first,
                    std::vector<Fr>& second,
                    size_t first_size,
                    size_t same_size,            //矩阵乘法的vector重组
                    size_t last_size);



// ---------------- template function (header-only) ----------------
template <typename T>
T host_partial_me(T *points, uint inputsize, typename vector<Fr>::const_iterator begin, typename vector<Fr>::const_iterator end)
{
    if (begin >= end) return points[0];
    uint outputsize = (inputsize + 1) / 2;
    T *new_points = new T[outputsize];
    #pragma omp parallel for
    for (int i = 0; i < outputsize; i++) {
        uint id0 = i * 2;
        uint id1 = i * 2 + 1;
        if (id1 < inputsize)
            new_points[i] = points[id0] + *begin * (points[id1] - points[id0]);
        else
            new_points[i] = points[id0] - *begin * points[id0];
    }
    T result = host_partial_me(new_points, outputsize, begin + 1, end);
    delete[] new_points;
    return result;
}


