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

uint layer_num = 2;
uint SCALE = 1 << 16;
__global__ void equall(fr_t *a, fr_t *b, uint size){
    for(int i=0; i < size; i++){
        fr_t now = a[i] - b[i];
        assert(now.is_zero());
        
    }
}
__global__ void test_kernel(fr_t *a, fr_t *b, fr_t *c, uint size)
{
    fr_t now; 
    for(int i=0; i < 100; i++){
        now.zero();
        for(int j=0; j < size; j++){
            now = now + a[j] * b[j * 11008 + i];
        }
        now -= c[i];
        assert(now.is_zero());
        //printf("yes!\n");
    }
    
}

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

    test_kernel<<<1, 1>>>(x.gpu_data, up.gpu_data, y.gpu_data, 4096);
    CUDA_DEBUG;

    cout <<"--------------------------------------------------------------------" << endl;

    vector<Fr> u;
    u.resize(Log2(y_size));

    long tmp = x_size;
    tmp = (tmp * up_size) / y_size / layer_num;
    uint same_size = sqrt(tmp);
    printf("same_size: %d\n", same_size);
    vector<Fr> v;
    v.resize(Log2(same_size));
    cout << "u_size: " << u.size() << " " << "v_size: " << v.size() << endl;

    for(uint i = 0; i < u.size(); i++) u[i] = Fr::random_element();
    for(uint i = 0; i < Log2(same_size); i++) v[i] = Fr::random_element();

    fr_t *U , *V;
    cudaMalloc(&U, sizeof(fr_t) * u.size());
    cudaMemcpy(U, u.data(), sizeof(fr_t) * u.size(), cudaMemcpyHostToDevice);

    cout <<"--------------------------------------------------------------------" << endl;
    CUDA_TIMER_START(verify);

    uint y_dim2 = up_size / same_size / layer_num;
    cout << "y_dim2: " << y_dim2 << endl;

    // for(int i = 1; i < layer_num; i++){
    //     cudaMemcpy(y.gpu_data, y.gpu_data + 1024 * 11008 * i, sizeof(fr_t) * 1024 * 11008, cudaMemcpyDeviceToDevice);
    //     CUDA_DEBUG;
    //     cudaMemcpy(x.gpu_data, x.gpu_data + 1024 * 4096 * i, sizeof(fr_t) * 1024 * 4096, cudaMemcpyDeviceToDevice);
    //     CUDA_DEBUG;
    //     cudaMemcpy(up.gpu_data, up.gpu_data + 4096 * 11008 * i, sizeof(fr_t) * 4096 * 11008, cudaMemcpyDeviceToDevice);

    //     y.partial_me(y_size / layer_num, U, x_size / same_size / layer_num, Log2(y_dim2), y_dim2);
    //     y.partial_me(y_dim2, U, y_dim2, 0, 1);
    
    //     x.partial_me(x_size / layer_num, U, x_size / same_size / layer_num, Log2(y_dim2), same_size);
    
    //     up.partial_me(up_size / layer_num, U, y_dim2, 0, 1);            
        
    //     x.mul_with_size(same_size, up);

    //     CUDA_DEBUG;

    //     Fr *host_x = new Fr[same_size];
    //     cudaMemcpy(host_x, x.gpu_data, sizeof(Fr) * same_size, cudaMemcpyDeviceToHost);
    //     Fr now = Fr::zero();
    //     for(int i = 0; i < same_size; i++) now += host_x[i];
    //     Fr now1;
    //     cudaMemcpy(&now1, y.gpu_data, sizeof(Fr), cudaMemcpyDeviceToHost);  
    //     assert(now == now1);

    //     CUDA_DEBUG;
    //     CUDA_TIMER_STOP(verify);
        
    // }
    
    

    // y.partial_me(y_size, U, x_size / same_size, Log2(y_dim2), y_dim2);
    // y.partial_me(y_dim2, U, y_dim2, 0, 1);
    // x.partial_me(x_size, U, x_size / same_size, Log2(y_dim2), same_size);
    // up.partial_me(up_size, U, y_dim2, 0, 1);                            
    // up.partial_me(up_size / y_dim2, U, layer_num, Log2(y_size / layer_num), same_size);

    

    cout << Log2(y_size / layer_num) << endl;
    y.partial_me(y_size, U, layer_num, Log2(y_size / layer_num), y_size / layer_num);
    x.partial_me(x_size, U, layer_num, Log2(y_size / layer_num), x_size / layer_num);
    up.partial_me(up_size, U, layer_num, Log2(y_size / layer_num), up_size / layer_num);
    test_kernel<<<1, 1>>>(x.gpu_data, up.gpu_data, y.gpu_data, 4096);
    
    

    // x.mul_with_size(same_size, up);

    // CUDA_DEBUG;

    // Fr *host_x = new Fr[same_size];
    // cudaMemcpy(host_x, x.gpu_data, sizeof(Fr) * same_size, cudaMemcpyDeviceToHost);
    // Fr now = Fr::zero();
    // for(int i = 0; i < same_size; i++) now += host_x[i];
    // Fr now1;
    // cudaMemcpy(&now1, y.gpu_data, sizeof(Fr), cudaMemcpyDeviceToHost);  
    // assert(now == now1);

    // CUDA_DEBUG;
    // CUDA_TIMER_STOP(verify);


}