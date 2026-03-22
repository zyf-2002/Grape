#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include "fr-tensor.cuh"
#include "Hyrax.cuh"
#include "ec_operation.cuh"
#include "timer.cuh"
#include <omp.h>

using namespace std;
using namespace libsnark;

uint npoints = 16384;
uint W = 1 << 8;  //pre_windows
uint input_tokens = 2048;
uint layer_scale = 1;

int main(int argc, char* argv[])
{
    ppT::init_public_params();
    affine_t *points;
    cudaMalloc((void **)&points, sizeof(affine_t) * npoints * W * layer_scale);
    auto cpu_points = precompute_generators(npoints * layer_scale, W, points);

    savebin("../data/points.bin", points, sizeof(affine_t) * npoints * W * layer_scale);
    savebin("../data/cpu_points.bin", cpu_points.data(), cpu_points.size() * sizeof(bn128), false);
    CUDA_DEBUG;

}