#include <iostream>
#include <chrono>
#include <vector>
#include "fr-tensor.cuh"
#include "Hyrax.cuh"
#include "g1-tensor.cuh"
#include "msm.cuh"
#include <omp.h>

using namespace std;
using namespace libsnark;

const size_t N = 11008;
const size_t W = 1 << 8;
const size_t layer_num = 4;


int main(int argc, char *argv[])
{
    
    ppT::init_public_params();
    affine_t *out;
    cudaMalloc((void **)&out, sizeof(affine_t) * N * W);
    auto cpu_pre = precompute_generators(N, W, out);

    int *int_data;
    string filename = "../data/X_up_rem.bin";

    CPU_TIMER_START(load_data);
    auto size = findsize(filename) / sizeof(int);
    cout << "size: " << size << endl;   
    cudaMalloc((void **)&int_data, sizeof(int) * size);
    loadbin(filename, int_data, sizeof(int) * size);
    CUDA_DEBUG;

    CPU_TIMER_STOP(load_data);


    jacob_t *commitment;
    uint outputsize = size / N;
    cudaMalloc((void **)&commitment, sizeof(jacob_t) * outputsize);

    Hyrax hyrax(size, N, int_data, out, commitment, cpu_pre[0]);

    CUDA_TIMER_START(Hyrax_commit);
    hyrax.commit();
    CUDA_DEBUG;
    CUDA_TIMER_STOP(Hyrax_commit); 

    uint x_num = Log2(size);
    vector<Fr> cpu_x;
    cpu_x.resize(x_num);
    #pragma omp parallel for
    for(int i = 0; i < x_num; i++){
        cpu_x[i] = Fr::random_element();
    }
    

    fr_t *x;
    cudaMalloc((void **)&x, sizeof(fr_t) * x_num);

    cudaMemcpy(x, cpu_x.data(), sizeof(fr_t) * x_num, cudaMemcpyHostToDevice);

    CUDA_DEBUG;

    
    CUDA_TIMER_START(Hyrax_eval);
    auto claim = hyrax.eval(x);
    CUDA_DEBUG;
    CUDA_TIMER_STOP(Hyrax_eval); 

   
    Fr c = Fr::random_element();
    fr_t *c_dev; cudaMalloc(&c_dev, sizeof(fr_t)); cudaMemcpy(c_dev, &c, sizeof(fr_t), cudaMemcpyHostToDevice);


    CUDA_TIMER_START(Hyrax_open)
    auto proof = hyrax.open(c_dev);
    CUDA_DEBUG;
    CUDA_TIMER_STOP(Hyrax_open)





    CPU_TIMER_START(verify);

    bn128 *host_commitment = new bn128[outputsize];
    cudaMemcpy(host_commitment, commitment, sizeof(jacob_t) * outputsize, cudaMemcpyDeviceToHost);
    uint start_x = Log2(N);
    bn128 *CC = new bn128[layer_num];
    bn128 C_;

    CPU_TIMER_START(partial);
    for(int i = 0; i < layer_num; i++){
        CC[i] = host_partial_me<bn128>(host_commitment + i * (size / layer_num / N), size / N / layer_num, cpu_x.begin() + start_x, cpu_x.end() - Log2(layer_num));
    }
    C_ = host_partial_me<bn128>(CC, layer_num, cpu_x.begin() + Log2(size / layer_num), cpu_x.end());
    CPU_TIMER_STOP(partial);
    
    
    bn128 now = bn128::zero();

    CPU_TIMER_START(partiall);
    #pragma omp parallel
    {
        bn128 local = bn128::zero();
        #pragma omp for
        for (int j = 0; j < N; j++) {
            local = local + proof.z[j] * cpu_pre[j];
        }
        #pragma omp critical
        now = now + local;
    }
    CPU_TIMER_STOP(partiall);

    C_ = c * C_;
    C_ = C_ + proof.com_d;
    assert(C_ == now);
    

    Fr az;
    
    az = host_partial_me<Fr>(proof.z, N, cpu_x.begin(), cpu_x.begin() + start_x);
    
    bn128 left = (c * proof.result) * cpu_pre[0] + proof.com_s;
    bn128 right = az * cpu_pre[0];
    assert(left == right);
    
    
    cudaFree(c_dev);

    CPU_TIMER_STOP(verify);




    // int* cpu_data = new int[size];
    // cudaMemcpy(cpu_data, int_data, sizeof(int) * size, cudaMemcpyDeviceToHost);
    

    //Fr *cpu_scalars = new libff::alt_bn128_Fr[size];

    // auto start1 = std::chrono::high_resolution_clock::now();

    // CPU_TIMER_START(test);
    
    // #pragma omp parallel for
    // for(int i=0; i < size; i++){
    //     cpu_scalars[i] = Fr(long(cpu_data[i]));
    // }
    // CPU_TIMER_STOP(test);

    // auto end1 = std::chrono::high_resolution_clock::now();
    // double ms = std::chrono::duration<double, std::milli>(end1 - start1).count();

    // std::cout << "执行时间: " << ms << " ms" << std::endl;

    // bn128* gpu_commit = new bn128[outputsize];
    // cudaMemcpy(gpu_commit, commitment, sizeof(jacob_t) * outputsize, cudaMemcpyDeviceToHost);

    // #pragma omp parallel for
    // for(int i=0; i < size; i+= N){
    //     bn128 now = bn128::zero();
    //     for(int j=0; j < N; j++){
    //         now = now + cpu_scalars[i + j] * cpu_pre[j];
    //     }
    //     assert(now == gpu_commit[i / N]);
    //     //printf("hello\n");
    // }

    // CUDA_DEBUG;

    // CUDA_TIMER_START(int_to_fr);
    // fr_t *value;
    // cudaMalloc((void **)&value, sizeof(fr_t) * size);
    // int_to_fr<<<(size + 512 - 1) / 512, 512>>>(int_data, value, size);
    // CUDA_DEBUG;
    // CUDA_TIMER_STOP(int_to_fr);

    


    // cudaFree(commitment);
    // cudaFree(int_data);
    // cudaFree(out);
    // cudaFree(value);
  


    return 0;
}