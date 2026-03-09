#include <iostream>
#include <chrono>
#include <vector>
#include "fr-tensor.cuh"
#include "Hyrax.cuh"
#include "ec_operation.cuh"
#include "prove.cuh"
#include "tlookup.cuh"
#include <omp.h>

using namespace std;
using namespace libsnark;
uint lower = - (1 << 15);
uint SCALE = 1 << 16;
uint npoints = 16384;
const size_t W = 1 << 8;  //pre_windows
int main(int argc, char* argv[])
{
    if (argc < 2) {
        return 1;
    }
    uint layer_num = atoi(argv[1]);
    cout << "layer_num: " << layer_num << endl;

    ppT::init_public_params();

    affine_t *points;
    cudaMalloc((void **)&points, sizeof(affine_t) * npoints * W * layer_num);
    auto cpu_points = precompute_generators(npoints * layer_num, W, points);
    Hyrax hyrax(layer_num, npoints, points, cpu_points[0]);
    CUDA_DEBUG;


    CPU_TIMER_START(read_data);
    
    int *x_data = nullptr;
    uint x_size = load_data("../data/R/up_out-7.bin", &x_data);
    x_size /= (4 / layer_num);
    uint pad_size = 1 << Log2(x_size);
    CPU_TIMER_STOP(read_data);
    
    cout <<"--------------------------------------------------------------------" << endl;

    tLookupRange rs(lower, SCALE);

    
    FrTensor x_tensor(layer_num * 2048 * 16384);
    FrTensor tlookup_A(layer_num * 2048 * 16384);
    
    for(int i = 0; i < 1; i++){
        x_tensor.get_data(x_size, x_data);
        auto v = random_vec(Log2(x_size));
        CPU_TIMER_START(test);
        Fr value = rs.prove(x_tensor, tlookup_A, v, hyrax, 16384);
        CUDA_DEBUG;
        CPU_TIMER_STOP(test);
    }
    
    cudaFree(x_data);

    cout <<"--------------------------------------------------------------------" << endl;
   
}