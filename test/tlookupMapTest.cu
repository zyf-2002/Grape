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
// int lower = - (1 << 18);
int lower = - (1 << 20) + 1;
// uint length = 1 << 19;
uint length = 1 << 20;
uint npoints = 16384;
uint com_scale = 16;
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
    cudaMalloc((void **)&points, npoints * com_scale * sizeof(affine_t) * W);
    loadbin("../data/points.bin", points, npoints * com_scale * sizeof(affine_t) * W);
    vector<bn128> cpu_points(npoints * com_scale);
    loadbin("../data/cpu_points.bin", cpu_points.data(), npoints * com_scale * sizeof(bn128), false);
    CUDA_DEBUG;

    Hyrax hyrax(layer_num, npoints, points, cpu_points[0]);

    CPU_TIMER_START(read_data);
    
    int *in_data = nullptr;
    //uint in_size = load_data("../data/table/swiglu_input.bin", &in_data);
    uint in_size = load_data("../data/Q/exp_input-7.bin", &in_data);
    int *out_data = nullptr;
    //uint out_size = load_data("../data/table/swiglu_output.bin", &out_data);
    uint out_size = load_data("../data/Q/exp_output-7.bin", &out_data);
    in_size /= (4 /layer_num);
    out_size /= (4 /layer_num);
    uint pad_size = 1 << Log2(in_size);
    CPU_TIMER_STOP(read_data);
    
    cout <<"--------------------------------------------------------------------" << endl;

    // tLookupRangeMapping swiglu(lower, length, "../data/table/swiglu-table.bin");
    tLookupRangeMapping exp(lower, length, "../data/table/exp-table.bin");
    
    FrTensor s_in(layer_num * 2048 * 16384);
    FrTensor s_out(layer_num * 2048 * 16384);
    FrTensor A(layer_num * 2048 * 16384);

    s_in.get_data(in_size, in_data);
    s_out.get_data(out_size, out_data);

    double com_time = 0;
    CPU_TIMER_START(prove);
    auto v = random_vec(Log2(in_size));
    exp.prove(s_in, s_out, A, v, hyrax, 2048, com_time);
    CUDA_DEBUG;
    CPU_TIMER_STOP(prove);
    cout <<"--------------------------------------------------------------------" << endl;
    cudaFree(in_data);
    cudaFree(out_data);
}