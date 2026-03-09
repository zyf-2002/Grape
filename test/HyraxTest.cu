#include <iostream>
#include <chrono>
#include <vector>
#include "fr-tensor.cuh"
#include "Hyrax.cuh"
#include "ec_operation.cuh"
#include <omp.h>

using namespace std;
using namespace libsnark;

const uint N = 1024;   
const uint W = 1 << 8;  //pre_windows
const uint layer_num = 4;
uint npoints = 16384;

int main(int argc, char *argv[])
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
    CUDA_DEBUG;

    cout<<"----------------------------------"<<"precompute done"<< "----------------------------------"<<endl;

    int *tensor = nullptr;
    string filename = "../data/W/NormSecond-7.bin";
    CPU_TIMER_START(load_data);
    uint size = load_data(filename, &tensor);

    //auto size = findsize(filename) / sizeof(int);
    //generate_data(&tensor, size);
    size = size / (4 / layer_num);
    CUDA_DEBUG;
    CPU_TIMER_STOP(load_data);

    cout<<"----------------------------------"<<"load_data done"<< "----------------------------------"<<endl;

    Hyrax hyrax(layer_num, npoints, points, cpu_points[0]);
    CUDA_DEBUG;
    FrTensor fr_tensor(size, tensor);
    // FrTensor t1(1 << Log2(size));   FrTensor t2(1 << Log2(size));
    // CUDA_DEBUG;
    // t1.get_data(size, tensor);
    // CUDA_DEBUG;
    // t1.pad(t2, N, 1 << Log2(N), t2.size, Fr::zero());
    // printf("hello\n");
    // CUDA_DEBUG;
    CPU_TIMER_START(commit);
    pad_int(&tensor, N, 1 << Log2(N), (size/N)*(1<<Log2(N)));
    jacob_t *commitment = hyrax.commit(tensor, (size/N)*(1<<Log2(N)), 1 << Log2(N));
    CUDA_DEBUG;
    CPU_TIMER_STOP(commit);

    // CPU_TIMER_START(commit_fr);
    // jacob_t *commitment = hyrax.commit(t1, 1 << Log2(size), 1 << Log2(N));
    // CUDA_DEBUG;
    // CPU_TIMER_STOP(commit_fr);

    // check_G_equal<<<(size / N / layer_num) / 128, 128>>>(commitment1, commitment, size / N / layer_num);

    cout<<"----------------------------------"<<"commit done"<< "----------------------------------"<<endl;

    vector<Fr> eval_point;
    eval_point.resize(Log2(size));
    #pragma omp parallel for
    for(int i = 0; i < Log2(size); i++){
        eval_point[i] = Fr::random_element();
    }
    
    Fr c = Fr::random_element();
    CPU_TIMER_START(open);
    auto proof = hyrax.open(fr_tensor, eval_point, c, size, N);
    CUDA_DEBUG;
    CPU_TIMER_STOP(open);

    cout<<"----------------------------------"<<"open_proof done"<< "----------------------------------"<<endl;


    CPU_TIMER_START(verify);
    bn128 *host_commitment = new bn128[size / N / layer_num];
    cudaMemcpy(host_commitment, commitment, sizeof(jacob_t) * (size / N / layer_num), cudaMemcpyDeviceToHost);
    uint start_x = Log2(N * layer_num);
    bn128 *layer_C_ = new bn128[layer_num];
    bn128 C_;

    
    for(int i = 0; i < layer_num; i++){
        layer_C_[i] = host_partial_me<bn128>(host_commitment + i * (size / layer_num / N / layer_num), size / N / layer_num / layer_num, eval_point.begin() + start_x, eval_point.end() - Log2(layer_num));
    }
    C_ = host_partial_me<bn128>(layer_C_, layer_num, eval_point.begin() + Log2(size / layer_num), eval_point.end());
    
    
    
    bn128 commit_z = bn128::zero();
    #pragma omp parallel
    {
        bn128 local = bn128::zero();
        #pragma omp for
        for (int j = 0; j < (1 << Log2(N)) * layer_num; j++) {
            local = local + proof.z[j] * cpu_points[j];
        }
        #pragma omp critical
        commit_z = commit_z + local;
    }

    C_ = c * C_;
    C_ = C_ + proof.commit_d;
    assert(C_ == commit_z);
    

    Fr az;

    az = host_partial_me<Fr>(proof.z, (1 << Log2(N)) * layer_num, eval_point.begin(), eval_point.begin() + start_x);
    
    bn128 left = (c * proof.result) * cpu_points[0] + proof.commit_ad;
    bn128 right = az * cpu_points[0];
    assert(left == right);
    
    CUDA_DEBUG;
    
    CPU_TIMER_STOP(verify);

    
    cout<<"----------------------------------"<<"verify done"<< "----------------------------------"<<endl;


    cudaFree(tensor);
    cudaFree(points);
    cudaFree(commitment);
    delete[] host_commitment;
    delete[] layer_C_;

    CUDA_DEBUG;

    cout<<"----------------------------------"<<"free_data done"<< "----------------------------------"<<endl;


    // int* cpu_data = new int[size];
    // cudaMemcpy(cpu_data, tensor, sizeof(int) * size, cudaMemcpyDeviceToHost);
    

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

    // bn128* gpu_commit = new bn128[size / N];
    // cudaMemcpy(gpu_commit, commitment, sizeof(jacob_t) * (size /N), cudaMemcpyDeviceToHost);

    // #pragma omp parallel for
    // for(int i=0; i < size; i+= N){
    //     bn128 now = bn128::zero();
    //     for(int j=0; j < N; j++){
    //         now = now + cpu_scalars[i + j] * cpu_points[j];
    //     }
    //     assert(now == gpu_commit[i / N]);
    //     //printf("hello\n");
    // }

    // CUDA_DEBUG;

    // CUDA_TIMER_START(int_to_fr);
    // fr_t *value;
    // cudaMalloc((void **)&value, sizeof(fr_t) * size);
    // int_to_fr<<<(size + 512 - 1) / 512, 512>>>(tensor, value, size);
    // CUDA_DEBUG;
    // CUDA_TIMER_STOP(int_to_fr);

    


    //cudaFree(commitment);
    //cudaFree(tensor);
    //cudaFree(points);
    //cudaFree(value);
  


    return 0;
}