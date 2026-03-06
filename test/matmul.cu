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
    int *up_data = nullptr;
    uint up_size = load_data("../data/W/NormFirst-7.bin", &up_data);
    up_size /= (4 / layer_num);
    int *x_data = nullptr;
    uint x_size = load_data("../data/Q/rmsFirst-7.bin", &x_data);
    x_size /= (4 / layer_num);
    int *y_data = nullptr;
    uint y_size = load_data("../data/Q/up_out-7.bin", &y_data);
    y_size /= (4 / layer_num);
    int *y_rem_data = nullptr;
    uint y_rem_size = load_data("../data/R/up_out-7.bin", &y_rem_data);
    y_rem_size /= (4 / layer_num);
    CUDA_DEBUG;
    cout << "up_size: " << up_size << " " << "x_size: " << x_size << " " << "y_size: " << y_size << " " << "y_rem_size: " << y_rem_size << endl;
    CPU_TIMER_STOP(read_data);

    cout <<"--------------------------------------------------------------------" << endl;
    // CUDA_TIMER_START(int_to_fr);
    // FrTensor up(up_size, up_data);
    // FrTensor y_rem(y_rem_size, y_rem_data);
    // FrTensor x(x_size, x_data);
    // FrTensor y(y_size, y_data); 
    // y *= SCALE;
    // y += y_rem;
    // CUDA_DEBUG;
    // CUDA_TIMER_STOP(int_to_fr);

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

    //SumcheckWorkTmp matmul_tmp(layer_num);
    long long total_duration = 0;
    FrTensor up(up_size);
    FrTensor y_rem(y_rem_size);
    FrTensor x(x_size);
    FrTensor y(y_size); 
    for(int i = 0; i < 1; i++){
        for(uint j = 0; j < (4 / layer_num); j++){
            
            int_to_fr<<<(up_size + 256 - 1) / 256, 256>>>(up_data + j * up_size, up.gpu_data, up_size);
            int_to_fr<<<(y_size + 256 - 1) / 256, 256>>>(y_data + j * y_size, y.gpu_data, y_size);
            int_to_fr<<<(x_size + 256 - 1) / 256, 256>>>(x_data + j * x_size, x.gpu_data, x_size);
            int_to_fr<<<(y_rem_size + 256 - 1) / 256, 256>>>(y_rem_data + j * y_rem_size, y_rem.gpu_data, y_rem_size);
            CUDA_DEBUG;
            
            y *= SCALE;
            y += y_rem;


            uint y_dim2 = up_size / same_size / layer_num;
            cout << "y_dim2: " << y_dim2 << endl;

            y.partial_eval(y_size, U, y_size / y_dim2, Log2(y_dim2), y_dim2);
            y.partial_eval(y_dim2, U, y_dim2, 0, 1);
        
            Fr claim;
            cudaMemcpy(&claim, y.gpu_data, sizeof(Fr), cudaMemcpyDeviceToHost);  
            auto start = std::chrono::high_resolution_clock::now();
            prove_matmul(x, up, layer_num, 
                    y.size / layer_num / y_dim2, y_dim2, 
                        same_size, u, v, claim);
            CUDA_DEBUG;
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            total_duration += duration.count();
        }
    }

    printf("prove_matmul函数执行总时间: %.3f ms\n", total_duration / 1000.0);
    

    // CPU_TIMER_START(prove_matmul);
    // prove_matmul(x, up, matmul_tmp, layer_num, y.size / layer_num / y_dim2, y_dim2, same_size, u, v, claim);
    // CUDA_DEBUG;
    // CPU_TIMER_STOP(prove_matmul);


    

    // Fr *host_x = new Fr[same_size];
    // cudaMemcpy(host_x, x.gpu_data, sizeof(Fr) * same_size, cudaMemcpyDeviceToHost);
    // Fr now = Fr::zero();
    // for(int i = 0; i < same_size; i++) now += host_x[i];
    // Fr now1;
    // cudaMemcpy(&now1, y.gpu_data, sizeof(Fr), cudaMemcpyDeviceToHost);  
    // assert(now == now1);

    // CUDA_DEBUG;
    CUDA_TIMER_STOP(verify);


    cudaFree(x_data);
    cudaFree(y_data);
    cudaFree(up_data);
    cudaFree(y_rem_data);
    cudaFree(U);
    cudaFree(V);
    //delete [] host_x;

}