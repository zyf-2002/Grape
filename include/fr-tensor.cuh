#pragma once
#include "./field/alt_bn128.cuh"
#include "./ioutils.cuh"
#include "msm.cuh"
using namespace alt_bn128;

__device__ __forceinline__ fr_t shfl_down_Fr(
    const fr_t &a,
    unsigned int offset,
    int width,
    unsigned mask)
{
    fr_t r;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        r[i] = __shfl_down_sync(mask, a[i], offset, 32);
    }
    return r;
}


__global__ void int_to_fr(int *input, fr_t *out, size_t N){
    uint global_id = GET_GLOBAL_ID();
    if(global_id >= N) return;
    
    bool flag = input[global_id] < 0 ? true : false;
    uint tmp = abs(input[global_id]);
    out[global_id].set_uint(tmp);
    out[global_id].cneg(flag);
    out[global_id].to();
}

__global__ void Memory_alignment(fr_t *t, uint gap){
    uint gid = GET_GLOBAL_ID();
    t[gid] = t[gid * gap];
}

__global__ void Fr_partial_me_kernel(fr_t *t, fr_t *x, uint dim, uint other_dims, uint in_cur_dim, uint out_cur_dim, uint id, uint window_size, uint gap){
    uint gid = GET_GLOBAL_ID();
    if (gid >= other_dims * out_cur_dim * window_size) return;

    uint ind0 = gid / (out_cur_dim * window_size);
    uint ind1 = (gid / window_size) % out_cur_dim;
    uint ind2 = gid % window_size;
    
    uint gid0 = ind0 * dim + (2 * ind1) * gap + ind2;

    if (2 * ind1 + 1 < in_cur_dim) 
    {
        uint gid1 = ind0 * dim + (2 * ind1 + 1) * gap + ind2;
        t[gid0] = t[gid0] + x[id] * (t[gid1] - t[gid0]);
    }
    else 
    {
        t[gid0] = t[gid0] - x[id] * t[gid0];
    }

    // if(gid >= out_size) return;

    // uint gap_id = gid / window_size;
    // uint idx_in_gap = gid % window_size;
    // uint gid0 = 2 * gap_id * gap + idx_in_gap;
    // uint gid1 = (2 * gap_id + 1) * gap + idx_in_gap;


    // if (gid1 % dim >= window_size) t[gid0] = t[gid0] + x[id] * (t[gid1] - t[gid0]);
    // else t[gid0] = t[gid0] - t[gid0] * x[id];

}

__global__ void Fr_partial_me_kernel1(fr_t *t, fr_t *out, fr_t *x, uint in_size, uint out_size, uint id, uint gap){
    uint gid = GET_GLOBAL_ID();
    if(gid >= out_size) return;

    uint gap_id = gid / gap;
    uint idx_in_gap = gid % gap;
    uint gid0 = 2 * gap_id * gap + idx_in_gap;
    uint gid1 = (2 * gap_id + 1) * gap + idx_in_gap;

    if (gid1 < in_size) out[gid] = t[gid0] + x[id] * (t[gid1] - t[gid0]);
    else if (gid0 < in_size) out[gid] = t[gid0] - t[gid0] * x[id];
    else  out[gid].zero();

}

__global__ void Fr_elementwise_add(fr_t* arr1, fr_t* arr2, fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = arr1[gid] + arr2[gid];
}

__global__ void Fr_elementwise_mul(fr_t* arr1, fr_t* arr2, fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = arr1[gid] * arr2[gid];
}

__global__ void Fr_broadcast_mul(fr_t* arr, fr_t *x, fr_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = arr[gid] * x[0];
}


class FrTensor
{   
    public:
    fr_t* gpu_data;
    uint size;
    FrTensor(uint size);
    FrTensor(const FrTensor& t);
    
    ~FrTensor();
    FrTensor(uint size, int* cpu_data);
    FrTensor& operator=(const FrTensor& x);
    void partial_me(uint N, fr_t *x, uint c, uint id, uint gap);
    FrTensor partial_me1(fr_t *x, uint id, uint gap);
    void mul_with_size(uint sz, const FrTensor& t);
    FrTensor& operator+=(const FrTensor& t);
    FrTensor& operator*=(const FrTensor& t);
    FrTensor& operator*=(fr_t *x);
    FrTensor& operator*=(const uint x);
};

FrTensor& FrTensor::operator*=(fr_t *x)
{
    Fr_broadcast_mul<<<(size + 512 -1) / 512, 512>>>(gpu_data, x, gpu_data, size);
    cudaDeviceSynchronize();
    return *this;
}

FrTensor& FrTensor::operator*=(const uint x)
{
    Fr host_x = Fr(long(x));
    fr_t *d_x;
    cudaMalloc((void **)&d_x, sizeof(fr_t));
    cudaMemcpy(d_x, &host_x, sizeof(fr_t), cudaMemcpyHostToDevice);
    Fr_broadcast_mul<<<(size + 512 -1) / 512, 512>>>(gpu_data, d_x, gpu_data, size);
    cudaDeviceSynchronize();
    return *this;
}


FrTensor::~FrTensor()
{
   cudaFree(gpu_data);
    //cudaFreeAsync(gpu_data, 0);
    gpu_data = nullptr;
}

FrTensor& FrTensor::operator+=(const FrTensor& t)
{
    if (size != t.size) throw std::runtime_error("Incompatible dimensions");
    Fr_elementwise_add<<<(size + 512 - 1) / 512 , 512>>>(gpu_data, t.gpu_data, gpu_data, size);
    cudaDeviceSynchronize();
    return *this;
}

void FrTensor::mul_with_size(uint sz, const FrTensor& t)
{
    Fr_elementwise_mul<<<(sz + 512 - 1) / 512 , 512>>>(gpu_data, t.gpu_data, gpu_data, sz);
    cudaDeviceSynchronize();
}

FrTensor& FrTensor::operator*=(const FrTensor& t)
{
    if (size != t.size) throw std::runtime_error("Incompatible dimensions");
    Fr_elementwise_mul<<<(size + 512 - 1) / 512 , 512>>>(gpu_data, t.gpu_data, gpu_data, size);
    cudaDeviceSynchronize();
    return *this;
}

FrTensor::FrTensor(const FrTensor& t): size(t.size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(fr_t) * size);
    cudaMemcpy(gpu_data, t.gpu_data, sizeof(fr_t) * size, cudaMemcpyDeviceToDevice);
}


FrTensor& FrTensor::operator=(const FrTensor& t)
{
    //if (size != t.size) throw std::runtime_error("operator=: Incompatible dimensions");
    cudaMemcpy(gpu_data, t.gpu_data, sizeof(fr_t) * size, cudaMemcpyDeviceToDevice);
    return *this;
}

FrTensor Fr_partial_me1(const FrTensor& t, fr_t *x, uint id, uint gap)
{
    if (t.size == gap) return t; 
    uint num_gaps = (t.size  + 2 * gap - 1) / (2 * gap);
    uint out_size = num_gaps * gap;
    //CUDA_TIMER_START(malloc);
    FrTensor t_new(out_size);
    CUDA_DEBUG;
    printf("%u\n",id);

    //CUDA_TIMER_STOP(malloc);
    
    Fr_partial_me_kernel1<<<(out_size + 512 -1) / 512, 512>>>(t.gpu_data, t_new.gpu_data, x, t.size, out_size, id, gap);
    
    cudaDeviceSynchronize();
    //CUDA_DEBUG;
    return Fr_partial_me1(t_new, x, id + 1, gap);
}

void Fr_partial_me(uint N, const FrTensor& t, fr_t *x, uint id, uint cur_dim, uint window_size)
{
    uint out_size = N;
    uint gap = window_size;
    uint count = Log2(cur_dim);
    uint cur_dim_in = cur_dim;
    uint dim = cur_dim * window_size;
    
    while(count--){
        uint cur_dim_out = (cur_dim_in + 1) / 2;
        uint other_dims = out_size / (cur_dim_in * window_size);
        out_size = other_dims * cur_dim_out * window_size;

        Fr_partial_me_kernel<<<(out_size + 512 -1) / 512, 512>>>(t.gpu_data, x, dim, other_dims, cur_dim_in, cur_dim_out, id, window_size, gap);
        cudaDeviceSynchronize();
        id += 1;
        gap = gap << 1;
        cur_dim_in = cur_dim_out;
    }
    return;
}

FrTensor FrTensor::partial_me1(fr_t *x, uint id, uint gap)
{
    return Fr_partial_me1(*this, x, id, gap);
}

void FrTensor::partial_me(uint N, fr_t *x, uint c, uint id, uint gap)  //N是需要操作的空间长度，N不一定等于size
{
    Fr_partial_me(N, *this, x, id, c, gap);
    CUDA_DEBUG;
    CUDA_TIMER_START(move);
    if(gap == 1){
        uint thread = (N / c) > 512 ? 512 : (N / c);
        Memory_alignment<<<((N / c) + 512 -1) / 512, thread>>>(gpu_data, c);
    }
    else{
        for(int i = 1; i < (N / gap / c); i++){
            //printf("%d\n", i);
            cudaMemcpy(gpu_data + i * gap, gpu_data + i * c * gap , sizeof(fr_t) * gap, cudaMemcpyDeviceToDevice);
        }
    }
    CUDA_DEBUG;
    CUDA_TIMER_STOP(move);
}

FrTensor::FrTensor(uint size): size(size), gpu_data(nullptr)
{
    //cudaMallocAsync(&gpu_data, sizeof(fr_t) * size, 0);
    cudaMalloc((void **)&gpu_data, sizeof(fr_t) * size);
}

FrTensor::FrTensor(uint size, int* int_data): size(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(fr_t) * size);
    CUDA_TIMER_START(inttos);
    int_to_fr<<<(size + 512 -1) / 512, 512>>>(int_data, gpu_data, size);
    CUDA_DEBUG;
    CUDA_TIMER_STOP(inttos); 
    
    cudaDeviceSynchronize();
}

