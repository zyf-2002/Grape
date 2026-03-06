#include <iostream>
#include <chrono>
#include <vector>
#include "fr-tensor.cuh"
#include "Hyrax.cuh"
#include "ec_operation.cuh"
#include "prove.cuh"
#include <omp.h>

using namespace std;
using namespace libsnark;

uint second_size = 11008;

uint SCALE = 1 << 16;

int main(int argc, char* argv[])
{
    if (argc < 2) {
        return 1;
    }
    uint layer_num = atoi(argv[1]);
    cout << "layer_num: " << layer_num << endl;

    ppT::init_public_params();

    CPU_TIMER_START(read_data);
    int *up_out_data = nullptr;
    uint up_out_size = load_data("../data/Q/up_out-7.bin", &up_out_data);
    up_out_size /= (4 / layer_num);
    int *silu_out_data = nullptr;
    uint silu_out_size = load_data("../data/Q/silu_out-7.bin", &silu_out_data);
    silu_out_size /= (4 / layer_num);
    int *upSilu_data = nullptr;
    uint upSilu_size = load_data("../data/Q/upSilu-7.bin", &upSilu_data);
    upSilu_size /= (4 / layer_num);
    int *upSilu_rem_data = nullptr;
    uint upSilu_rem_size = load_data("../data/R/upSilu-7.bin", &upSilu_rem_data);
    upSilu_rem_size /= (4 / layer_num);
    CUDA_DEBUG;
    cout << "up_out_size: " << up_out_size << " " << "silu_out_size: " << silu_out_size << " " << "upSilu_size: " << upSilu_size << endl;
    CPU_TIMER_STOP(read_data);

    cout <<"--------------------------------------------------------------------" << endl;

    CUDA_TIMER_START(int_to_fr);

    FrTensor up_out(up_out_size, up_out_data);
    FrTensor upSilu_rem(upSilu_rem_size, upSilu_rem_data);
    FrTensor silu_out(silu_out_size, silu_out_data);
    FrTensor upSilu(upSilu_size, upSilu_data);
    
    upSilu *= SCALE;
    upSilu += upSilu_rem;
    CUDA_DEBUG;
    CUDA_TIMER_STOP(int_to_fr);

    cout <<"--------------------------------------------------------------------" << endl;

    vector<Fr> u;
    u.resize(Log2(upSilu_size));
    vector<Fr> v;
    v.resize(Log2(upSilu_size));
    cout << "u_size: " << u.size() << " " << "v_size: " << v.size() << endl;

    for(uint i = 0; i < u.size(); i++) u[i] = Fr::random_element();
    for(int i = 0; i < v.size(); i++) v[i] = Fr::random_element();

    fr_t *U , *V;
    cudaMalloc(&U, sizeof(fr_t) * u.size());
    cudaMalloc(&V, sizeof(fr_t) * v.size());
    cudaMemcpy(U, u.data(), sizeof(fr_t) * u.size(), cudaMemcpyHostToDevice);


    cout <<"--------------------------------------------------------------------" << endl;
    CUDA_TIMER_START(verify);

    CPU_TIMER_START(table_init);
    upSilu.partial_eval(upSilu_size, U, upSilu_size / second_size, Log2(second_size), second_size);
    upSilu.partial_eval(second_size, U, second_size, 0, 1);
    CUDA_DEBUG;
    CPU_TIMER_STOP(table_init);

    Fr claim;
    cudaMemcpy(&claim, upSilu.gpu_data, sizeof(Fr), cudaMemcpyDeviceToHost);  

    //SumcheckWorkTmp eleMul_tmp(layer_num * 2048);
    CUDA_DEBUG;

    CPU_TIMER_START(prove_eleMul);
    prove_eleMul(up_out, silu_out, layer_num, 2048, second_size, u, v, claim);
    CUDA_DEBUG;
    CPU_TIMER_STOP(prove_eleMul);


    
    cudaFree(up_out_data);
    cudaFree(silu_out_data);
    cudaFree(upSilu_data);
    cudaFree(upSilu_rem_data);
    cudaFree(U);
    cudaFree(V);
    //delete [] host_x;

}