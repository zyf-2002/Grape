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


int main()
{
    uint layer_num = 4;
    
    ppT::init_public_params();
    

    int *tensor = nullptr;
    string filename = "../data/Q/ffn_input-7.bin";
    CPU_TIMER_START(load_data);
    uint size = load_data(filename, &tensor);
    cout << "size: " << size << endl;
    CUDA_DEBUG;
    CPU_TIMER_STOP(load_data);
    FrTensor X(size, tensor);
    auto u1 = random_vec(Log2(size)); 
    auto u2 = random_vec(Log2(size)); 
    Fr claim1 = X(u1);
    X.get_data(size, tensor);
    Fr claim2 = X(u2);
    X.get_data(size, tensor);
    auto v1 = random_vec(Log2(size));
    CPU_TIMER_START(combine);
    combine_claims(X, {claim1, claim2}, {u1, u2}, v1, 2, size);
    CUDA_DEBUG;
    CPU_TIMER_STOP(combine);


    cudaFree(tensor);
    

    CUDA_DEBUG;

    cout<<"----------------------------------"<<"free_data done"<< "----------------------------------"<<endl;

    return 0;
}