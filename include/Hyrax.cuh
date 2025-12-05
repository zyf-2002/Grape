#ifndef HYRAX_CUH
#define HYRAX_CUH
#include <iostream>
#include "ioutils.cuh"
#include "./field/alt_bn128.cuh"
#include "fr-tensor.cuh"
#include "msm.cuh"
using namespace alt_bn128;
using namespace std;

uint layer = 4;
__device__ __forceinline__ fp_t shfl_down_Fp(
    const fp_t &a,
    unsigned int offset,
    int width,
    unsigned mask)
{
    fp_t r;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        r[i] = __shfl_down_sync(mask, a[i], offset, 32);
    }
    return r;
}

__device__ __forceinline__ jacob_t shfl_down_G1(
    const jacob_t &p,
    unsigned int offset,
    int width,
    unsigned mask)
{
    fp_t X = shfl_down_Fp(p.getX(), offset, width, mask);
    fp_t Y = shfl_down_Fp(p.getY(), offset, width, mask);
    fp_t Z = shfl_down_Fp(p.getZ(), offset, width, mask);
    jacob_t r(X, Y, Z);
    return r;
}

__global__ void __launch_bounds__(512, 1) Hyrax_commit_int(int* scalars, affine_t *point, jacob_t *out, 
                                            uint N, uint npoints){

    assert(GET_TOTAL_THREADS() == (N / npoints) * 32);
    uint warp_id = GET_WARP_ID();
    uint lane_id = GET_LANE_ID();
    unsigned int mask = (1 << 8) - 1;

    out[warp_id].inf();
    
    for(int w = 3; w >=0; w--){
        jacob_t sum; sum.inf();
        for(int i = lane_id; i < npoints; i += 32){
            bool is_neg = scalars[warp_id * npoints + i] < 0 ? 1 : 0;
            uint s = abs(scalars[warp_id * npoints + i]);
            uint index = ((s >> (w * 8)) & mask) * npoints + i;
            affine_t tmp = point[index];
            if(index < npoints) assert(tmp.is_inf());
            tmp.cneg(is_neg);
            sum.add(tmp);
        }
        
        __syncwarp();
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum.add(shfl_down_G1(sum, offset, 32, 0xffffffffu));
        }
        
        __syncwarp();

        if(lane_id == 0){
            #pragma unroll
            for(int bit = 0; bit < 8; bit++){
                out[warp_id].dbl();
            }
            if(w == 3) assert(out[warp_id].is_inf());
            out[warp_id].add(sum);
        }
        
    }
    __syncthreads();
}

__global__ void __launch_bounds__(512, 1) Hyrax_commit_fr(fr_t* scalars, affine_t *point, jacob_t *out, 
                                            uint npoints){

    uint warp_id = GET_WARP_ID();
    uint lane_id = GET_LANE_ID();
    unsigned int mask = (1 << 8) - 1;

    out[warp_id].inf();
    
    for(int w = 31; w >=0; w--){
        jacob_t sum; sum.inf();
        for(int i = lane_id; i < npoints; i += 32){
            fr_t s = scalars[warp_id * npoints + i];
            s.from();
            uint index = ((s[w / 4] >> ((w % 4) * 8)) & mask) * npoints + i;
            affine_t tmp = point[index];
            //if(index < npoints) assert(tmp.is_inf());
            sum.add(tmp);
        }
        
        __syncwarp();
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum.add(shfl_down_G1(sum, offset, 32, 0xffffffffu));
        }
        
        __syncwarp();

        if(lane_id == 0){
            #pragma unroll
            for(int bit = 0; bit < 8; bit++){
                out[warp_id].dbl();
            }
            //if(w == 31) assert(out[warp_id].is_inf());
            out[warp_id].add(sum);
        }
        
    }
    __syncthreads();
}

__global__ void affine_to_jacob(jacob_t *d, affine_t *h, uint size){
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    d[tid] = h[tid + size];
    //assert(h[tid].is_inf());
}


class Hyrax_proof{
public:
    Fr result;
    Fr *z;
    bn128 com_d;
    bn128 com_s;


    Hyrax_proof(Fr result_, Fr* _z, bn128 _com_d, bn128 _com_s)
        : result(result_), z(_z), com_d(_com_d), com_s(_com_s){}
};

class Hyrax {
public:
    uint N;     //标量大小
    uint size;   //点大小（没有预计算之前的数量）
    int *scalar;
    FrTensor value;
    bn128 G;
    affine_t *g_affine;
    jacob_t *commitment;
    fr_t *X;

    Hyrax(uint n_, uint size_, int* s, affine_t * g, jacob_t * c, bn128 &G_)
        : N(n_), size(size_), scalar(s), g_affine(g), commitment(c), value(n_, s), G(G_){}

    

    ~Hyrax() { cudaFree(scalar); cudaFree(g_affine); cudaFree(commitment); cudaFree(X);}
    void commit();
    FrTensor eval(fr_t *x);
    Hyrax_proof open(fr_t *c);
};

Hyrax_proof Hyrax::open(fr_t *c){
    FrTensor tmp_value(N);
    tmp_value = value;

    Fr *host_d = new Fr[size];
    #pragma omp parallel for
    for(int i = 0; i < size; i++) host_d[i] = Fr::random_element();
    FrTensor d(size);
    cudaMemcpy(d.gpu_data, host_d, sizeof(fr_t) * size, cudaMemcpyHostToDevice);
    jacob_t *com_d;
    cudaMalloc(&com_d, sizeof(jacob_t));

    CUDA_TIMER_START(commit_d);
    Hyrax_commit_fr<<<1, 32>>>(d.gpu_data, g_affine, com_d, size);
    CUDA_DEBUG;
    CUDA_TIMER_STOP(commit_d);

    FrTensor a(size);

    CUDA_TIMER_START(calculate_a);
    value.partial_me(N, X, (N / layer) / size, Log2(size), size);

    value.partial_me(size * layer, X, layer, Log2(N / layer), size);
    a = value;
    CUDA_TIMER_STOP(calculate_a);
    
    Fr result;
    CUDA_TIMER_START(calculate_result);
    value.partial_me(size, X, size, 0, 1);
    CUDA_TIMER_STOP(calculate_result);
    cudaMemcpy(&result, value.gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);

    FrTensor z(size);
    z = a;
    z *= c;
    z += d;

    CUDA_TIMER_START(calculate_ad);
    d.partial_me(d.size, X, size, 0, 1);
   
    CUDA_TIMER_STOP(calculate_ad);
    
    Fr sum_ad;
    cudaMemcpy(&sum_ad, d.gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);
    bn128 com_ad;
    com_ad = sum_ad * G;
    
    Fr *host_z = new Fr[size];
    cudaMemcpy(host_z, z.gpu_data, sizeof(fr_t) * size, cudaMemcpyDeviceToHost);
    
    bn128 host_com_d;
    cudaMemcpy(&host_com_d, com_d, sizeof(bn128), cudaMemcpyDeviceToHost);

    Hyrax_proof proof(result, host_z, host_com_d, com_ad);

    value = tmp_value;
    printf("%u\n",size);
    value.partial_me(N, X, size, 0, 1);
    value.partial_me(N / size, X, 4, Log2(N / 4), 1024);
    value.partial_me(1024, X, 1024, Log2(size), 1);
    Fr kk;
    cudaMemcpy(&kk, value.gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);
    assert(kk == result);

    return proof;
}

FrTensor Hyrax::eval(fr_t *x) {
    X = x;
    // CUDA_TIMER_START(test);
    // FrTensor tmp(1);
    // tmp = value.partial_me1(x, 0, 1);
    // CUDA_DEBUG;
    // printf("%u\n",tmp.size);
    // CUDA_TIMER_STOP(test);  
    
    // value.partial_me(X, N / size , Log2(size), size);   
    // value.partial_me(X, size, 0, 1);


    // //result = value;
    // Fr *first = new Fr[1];
    // Fr *second = new Fr[1];
    // cudaMemcpy(first, tmp.gpu_data, sizeof(Fr), cudaMemcpyDeviceToHost);
    // cudaMemcpy(second, value.gpu_data, sizeof(Fr), cudaMemcpyDeviceToHost);
    
    // assert(first[0] == second[0]);
        
    FrTensor result(1);
    return result;
}

void Hyrax::commit() {
    uint thread = (N / size) * 32;
    Hyrax_commit_int<<<(thread + 512 - 1) / 512, 512>>>(scalar, g_affine, commitment, N, size);
}





    




#endif // HYRAX_CUH
