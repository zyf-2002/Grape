#ifndef HYRAX_CUH
#define HYRAX_CUH
#include <iostream>
#include "ioutils.cuh"
#include "./field/alt_bn128.cuh"
#include "fr-tensor.cuh"
#include "ec_operation.cuh"
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
    uint global_warp_id = GET_GLOBAL_WARP_ID();
    uint lane_id = GET_LANE_ID();
    unsigned int mask = (1 << 8) - 1;

    out[global_warp_id].inf();
    
    for(int w = 3; w >=0; w--){
        jacob_t sum; sum.inf();
        for(int i = lane_id; i < npoints; i += 32){
            bool is_neg = scalars[global_warp_id * npoints + i] < 0 ? 1 : 0;
            uint s = abs(scalars[global_warp_id * npoints + i]);
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
                out[global_warp_id].dbl();
            }
            if(w == 3) assert(out[global_warp_id].is_inf());
            out[global_warp_id].add(sum);
        }
        
    }
    __syncthreads();
}

__global__ void single_windows_sum(fr_t* scalars, affine_t *point, jacob_t *out, 
                                            uint npoints){
    uint lane_id = GET_LANE_ID();
    uint block_id = blockIdx.x;
    uint grid_size = gridDim.x;
    unsigned int mask = (1 << 8) - 1;
    jacob_t sum; sum.inf();
    for(int i = block_id; i < npoints; i += grid_size){
        fr_t s = scalars[i];
        s.from();
        uint index = (s[lane_id / 4] >> ((lane_id % 4) * 8)) & mask;
        sum.add(point[index * npoints + i]);
        __syncwarp();
    }

    out[block_id * 32 + lane_id] = sum;
}

__global__ void single_windows_reduce(jacob_t *in, jacob_t *out, uint num){
    uint lane_id = GET_LANE_ID();
    jacob_t sum; sum.inf();
    for(int i = 0; i < num; i++){
        sum.add(in[i * 32 + lane_id]);
    }
    __syncwarp();
    for (int offset = 1; offset < 32; offset <<= 1) {
        jacob_t other = shfl_down_G1(sum, offset, 32, 0xffffffffu);
        if (lane_id + offset < 32) {
            int shift_bits = offset * 8;     
            for(int i = 0; i < shift_bits; i++) other.dbl();
            sum.add(other);                  
        }
    }
    __syncwarp();
    if(lane_id == 0) out[0] = sum;

}

__global__ __launch_bounds__(128, 2) void many_windows_sum(fr_t* scalars, affine_t *point, jacob_t *out, uint npoints){
    uint lane_id = GET_LANE_ID();
    uint warp_id = GET_WARP_ID();
    unsigned int mask = (1 << 8) - 1;
    bucket_t sum; sum.inf();

    uint block_id = blockIdx.x;
    uint stride = blockDim.x / 32;

    for(int i = warp_id; i < npoints; i += stride){
        uint index = reinterpret_cast<uint8_t*>(&scalars[block_id * npoints + i])[lane_id];
        // uint s = scalars[blockIdx.x * npoints + i][lane_id / 4];
        // uint index = (s >> ((lane_id % 4) * 8)) & mask;
        sum.add(point[index * npoints + i]);
    }
    sum.to_jacobian();
    out[block_id * blockDim.x + threadIdx.x] = *reinterpret_cast<jacob_t*>(&sum);

}

__global__ void many_windows_reduce(jacob_t *in, jacob_t *out, uint num){
    uint lane_id = GET_LANE_ID();
    uint global_warp_id = GET_GLOBAL_WARP_ID();
    jacob_t sum; sum.inf();
    for(int i = 0; i < num; i++){
        sum.add(in[global_warp_id * 32 * num + i * 32 + lane_id]);
    }
    __syncwarp();

    for (int offset = 1; offset < 32; offset <<= 1) {
        jacob_t other = shfl_down_G1(sum, offset, 32, 0xffffffffu);
        if (lane_id + offset < 32) {
            int shift_bits = offset * 8;     
            for(int i = 0; i < shift_bits; i++) other.dbl();
            sum.add(other);                  
        }
    }

    __syncwarp();
    if(lane_id == 0) out[global_warp_id] = sum;

}

                                            


__global__ void affine_to_jacob(jacob_t *d, affine_t *h, uint size){
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    d[tid] = h[tid + size];
    //assert(h[tid].is_inf());
}


class Hyrax_proof{
public:
    size_t z_size; 
    Fr result;
    Fr *z;
    bn128 commit_d;
    bn128 commit_ad;


    Hyrax_proof(Fr result_, size_t z_len, bn128 commit_d_, bn128 commit_ad_)
        : result(result_), z_size(z_len),
          commit_d(commit_d_), commit_ad(commit_ad_) {

        z = new Fr[z_size];
    }

    ~Hyrax_proof() {
        delete[] z;
    }
};

class Hyrax {
public:
    uint size;     //标量大小
    uint N;   //点大小（没有预计算之前的数量）
    bn128 G;
    affine_t *g_affine;
    

    Hyrax(uint size_, uint N_, affine_t * g, bn128 &G_)
        : size(size_), N(N_), g_affine(g), G(G_){}

    ~Hyrax() {}

    jacob_t* commit(int *tensor);
    jacob_t* commit(FrTensor &tensor);
    //FrTensor eval(fr_t *x);
    Hyrax_proof open(int *tensor, vector<Fr> eval_point_, Fr c);
};

void single_commit(fr_t *d, affine_t *g, jacob_t *com, uint npoints){ 
    jacob_t *tmp_out;
    cudaMalloc(&tmp_out, sizeof(jacob_t) * 32 * 32);
    single_windows_sum<<<32, 32>>>(d, g, tmp_out, npoints);
    single_windows_reduce<<<1, 32>>>(tmp_out, com, 32);
    cudaFree(tmp_out);
}

Hyrax_proof Hyrax::open(int *tensor, vector<Fr> eval_point_, Fr c){
    FrTensor fr_tensor(size, tensor);
    // FrTensor tmp_value(size);
    // tmp_value = fr_tensor;

    fr_t *eval_point;
    cudaMalloc(&eval_point, sizeof(fr_t) * eval_point_.size());
    cudaMemcpy(eval_point, eval_point_.data(), sizeof(fr_t) * eval_point_.size(), cudaMemcpyHostToDevice);
    fr_t *c_dev; cudaMalloc(&c_dev, sizeof(fr_t)); cudaMemcpy(c_dev, &c, sizeof(fr_t), cudaMemcpyHostToDevice);

    Fr *host_d = new Fr[N];

    CPU_TIMER_START(open_gen_d);
    #pragma omp parallel for
    for(int i = 0; i < N; i++) host_d[i] = Fr::random_element();
    FrTensor d(N);
    CPU_TIMER_STOP(open_gen_d);

    cudaMemcpy(d.gpu_data, host_d, sizeof(fr_t) * N, cudaMemcpyHostToDevice);
    
    jacob_t *commit_d;
    cudaMalloc(&commit_d, sizeof(jacob_t));

    CUDA_TIMER_START(commit_d);
    //Hyrax_commit_fr<<<2, 128>>>(d.gpu_data, g_affine, commit_d, N / 2);
    single_commit(d.gpu_data, g_affine, commit_d, N);
    CUDA_DEBUG;
    CUDA_TIMER_STOP(commit_d);

    FrTensor x(N);

    CPU_TIMER_START(calculate_a);
    fr_tensor.partial_me(size, eval_point, (size / layer) / N, Log2(N), N);
    fr_tensor.partial_me(N * layer, eval_point, layer, Log2(size / layer), N);
    x = fr_tensor;
    CPU_TIMER_STOP(calculate_a);
    
    Fr result;
    CPU_TIMER_START(calculate_result);
    fr_tensor.partial_me(N, eval_point, N, 0, 1);
    CPU_TIMER_STOP(calculate_result);
    
    cudaMemcpy(&result, fr_tensor.gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);

    CPU_TIMER_START(calculate_z);
    FrTensor z(N);
    z = x;
    z *= c_dev;
    z += d;
    CPU_TIMER_STOP(calculate_z);

    CPU_TIMER_START(calculate_ad);
    d.partial_me(d.size, eval_point, N, 0, 1);
    CPU_TIMER_STOP(calculate_ad);
    
    Fr sum_ad;
    cudaMemcpy(&sum_ad, d.gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);
    bn128 commit_ad;
    commit_ad = sum_ad * G;
    
    bn128 host_com_d;
    cudaMemcpy(&host_com_d, commit_d, sizeof(bn128), cudaMemcpyDeviceToHost);

    Hyrax_proof proof(result, N, host_com_d, commit_ad);

    cudaMemcpy(proof.z, z.gpu_data, sizeof(fr_t) * N, cudaMemcpyDeviceToHost);
   
    // fr_tensor = tmp_value;
    // fr_tensor.partial_me(size, eval_point, N, 0, 1);
    // fr_tensor.partial_me(size / N, eval_point, 4, Log2(size / 4), 1024);
    // fr_tensor.partial_me(1024, eval_point, 1024, Log2(N), 1);
    // Fr kk;
    // cudaMemcpy(&kk, fr_tensor.gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);
    // assert(kk == result);

    cudaFree(c_dev);
    cudaFree(eval_point);
    cudaFree(commit_d);
    delete [] host_d;
    
    return proof;
}



jacob_t* Hyrax::commit(int *tensor) {
    jacob_t *commitment;
    cudaMalloc(&commitment, sizeof(jacob_t) * (size / N));

    uint thread = (size / N) * 32;
    Hyrax_commit_int<<<(thread + 512 - 1) / 512, 512>>>(tensor, g_affine, commitment, size, N);
    return commitment;
}

jacob_t* Hyrax::commit(FrTensor &tensor){ 
    jacob_t *commitment;
    cudaMalloc(&commitment, sizeof(jacob_t) * (size / N));
    uint threads_per_block = 128;
    jacob_t *tmp_out;
    cudaMalloc(&tmp_out, sizeof(jacob_t) * (size / N) * threads_per_block);
    tensor.from();
    CUDA_TIMER_START(window_sum);
    many_windows_sum<<<(size / N), threads_per_block>>>(tensor.gpu_data, g_affine, tmp_out, N);
    CUDA_TIMER_STOP(window_sum);

    uint blocknum = 32 * (size / N) / threads_per_block;
    CUDA_TIMER_START(window_reduce);
    many_windows_reduce<<<blocknum, threads_per_block>>>(tmp_out, commitment, threads_per_block / 32);
    CUDA_TIMER_STOP(window_reduce);

    tensor.to();
    cudaFree(tmp_out);
    return commitment;
}




#endif // HYRAX_CUH
