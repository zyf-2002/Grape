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

uint layer_num = 4;
uint SCALE = 1 << 16;

int main()
{
    ppT::init_public_params();

    CPU_TIMER_START(read_data);
    int *up_data;
    uint up_size = load_data("../data/up_layer.bin", &up_data);
    int *x_data;
    uint x_size = load_data("../data/X.bin", &x_data);
    int *y_data;
    uint y_size = load_data("../data/X_up.bin", &y_data);
    int *y_rem_data;
    uint y_rem_size = load_data("../data/X_up_rem.bin", &y_rem_data);
    CUDA_DEBUG;
    cout << "up_size: " << up_size << " " << "x_size: " << x_size << " " << "y_size: " << y_size << " " << "y_rem_size: " << y_rem_size << endl;
    CPU_TIMER_STOP(read_data);

    cout <<"--------------------------------------------------------------------" << endl;

    CUDA_TIMER_START(int_to_fr);
    FrTensor up(up_size, up_data);
    FrTensor x(x_size, x_data);
    FrTensor y(y_size, y_data); 
    FrTensor y_rem(y_rem_size, y_rem_data);
    y *= SCALE;
    y += y_rem;
    CUDA_DEBUG;
    CUDA_TIMER_STOP(int_to_fr);

    cout <<"--------------------------------------------------------------------" << endl;

    vector<Fr> u;
    u.resize(Log2(y_size));

    long tmp = x_size;
    tmp = (tmp * up_size) / y_size / layer_num;
    uint same_size = sqrt(tmp);
    vector<Fr> v;
    v.resize(Log2(same_size * layer_num));
    cout << "u_size: " << u.size() << " " << "v_size: " << v.size() << endl;

    for(uint i = 0; i < u.size(); i++) u[i] = Fr::random_element();
    for(int i = 0; i < v.size(); i++) v[i] = Fr::random_element();

    fr_t *U , *V;
    cudaMalloc(&U, sizeof(fr_t) * u.size());
    cudaMalloc(&V, sizeof(fr_t) * v.size());
    cudaMemcpy(U, u.data(), sizeof(fr_t) * u.size(), cudaMemcpyHostToDevice);


    cout <<"--------------------------------------------------------------------" << endl;
    CUDA_TIMER_START(verify);

    uint y_dim2 = up_size / same_size / layer_num;
    cout << "y_dim2: " << y_dim2 << endl;



    y.partial_eval(y_size, U, y_size / y_dim2, Log2(y_dim2), y_dim2);
    y.partial_eval(y_dim2, U, y_dim2, 0, 1);
    // x.partial_eval(x_size, U, x_size / same_size / layer_num, Log2(y_dim2), same_size);
    // up.partial_eval(up_size, U, y_dim2, 0, same_size);
    // x.mul_with_size(same_size * layer_num, up);
    // x.partial_eval(layer_num * same_size, U, layer_num, Log2(y_size / layer_num), same_size);

    Fr claim;
    cudaMemcpy(&claim, y.gpu_data, sizeof(Fr), cudaMemcpyDeviceToHost);  

    MatMulSumcheckWorkTmp matmul_tmp(layer_num);

    CPU_TIMER_START(prove_matmul);
    prove_matmul(x, up, matmul_tmp, layer_num, y.size / layer_num / y_dim2, y_dim2, same_size, u, v, claim);
    CUDA_DEBUG;
    CPU_TIMER_STOP(prove_matmul);

    
   
 
    // Fr *host_x = new Fr[same_size];
    // cudaMemcpy(host_x, x.gpu_data, sizeof(Fr) * same_size, cudaMemcpyDeviceToHost);
    // Fr now = Fr::zero();
    // for(int i = 0; i < same_size; i++) now += host_x[i];
    // Fr now1;
    // cudaMemcpy(&now1, y.gpu_data, sizeof(Fr), cudaMemcpyDeviceToHost);  
    // assert(now == now1);

    // CUDA_DEBUG;
    // CUDA_TIMER_STOP(verify);


    cudaFree(x_data);
    cudaFree(y_data);
    cudaFree(up_data);
    cudaFree(y_rem_data);
    cudaFree(U);
    cudaFree(V);
    //delete [] host_x;

}