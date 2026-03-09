#include "fr-tensor.cuh"

// 内核函数定义-----------------------------------------------------------------------------------------------------------
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

__global__ void pad_int_kernel(const int *input, int *output, uint N, uint pad_N, uint size){
    uint gid = GET_GLOBAL_ID();
    if(gid >= size) return;
    output[gid] =  gid % pad_N >= N ? 0 : input[(gid / pad_N) * N + gid % pad_N];   
}

__global__ void pad_fr_kernel(fr_t *input, fr_t *output, uint N, uint pad_N, uint size, fr_t *pad_val){ 
    uint gid = GET_GLOBAL_ID();
    if(gid >= size) return;
    output[gid] =  gid % pad_N < N ? input[(gid / pad_N) * N + gid % pad_N] : pad_val[0];
}

__global__ void int_to_fr(int *input, fr_t *out, size_t N){
    uint global_id = GET_GLOBAL_ID();
    if(global_id >= N) return;
    
    uint32_t flag = input[global_id] >> 31;          // 0 or 1
    uint32_t absx = abs(input[global_id]);
    out[global_id].set_uint(absx);
    out[global_id].cneg(flag);
    out[global_id].to();
}

__global__ void uint_to_fr(uint *input, fr_t *out, size_t N){
    uint global_id = GET_GLOBAL_ID();
    if(global_id >= N) return;
    out[global_id].set_uint(input[global_id]);
    out[global_id].to();
}

__global__ void Memory_alignment(fr_t *t, uint gap){
    uint gid = GET_GLOBAL_ID();
    t[gid] = t[gid * gap];
}

__global__ void Fr_partial_eval_kernel(fr_t *t, fr_t *x, uint dim, uint other_dims, uint in_cur_dim, uint out_cur_dim, uint id, uint window_size, uint gap){
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
}

__global__ void Fr_elementwise_add(fr_t* arr1, fr_t* arr2, fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = arr1[gid] + arr2[gid];
}

__global__ void Fr_elementwise_sub(fr_t* arr1, fr_t* arr2, fr_t* arr_out, uint n) 
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = arr1[gid] - arr2[gid];
}

__global__ void Fr_broadcast_add(fr_t* arr, fr_t *x, fr_t* arr_out, uint n)
{
  const uint gid = GET_GLOBAL_ID();
  if (gid >= n) return;
  arr_out[gid] = arr[gid] + x[0];
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

__global__ void from_kernel(fr_t *arr, uint size){
    uint gid = GET_GLOBAL_ID();
    if(gid >= size) return;
    arr[gid].from();
}

__global__ void to_kernel(fr_t *arr, uint size){
    uint gid = GET_GLOBAL_ID();
    if(gid >= size) return;
    arr[gid].to();
}

__global__ void inverse_kernel(fr_t *arr, uint size){
    uint gid = GET_GLOBAL_ID();
    if(gid >= size) return;
    arr[gid] = arr[gid].reciprocal();
}

__global__ void Fr_sum_reduction(fr_t *arr, fr_t *output, uint n) {
    extern __shared__ fr_t frsum_sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

    // Load input into shared memory
    frsum_sdata[tid] = (i < n) ? arr[i] : fr_t::get_zero();
    if (i + blockDim.x < n) frsum_sdata[tid] = frsum_sdata[tid] + arr[i + blockDim.x];

    __syncthreads();

    // Reduction in shared memory
    for (unsigned int offset = blockDim.x / 2; offset >= 32; offset >>= 1) {
        if (tid < offset) {
            frsum_sdata[tid] = frsum_sdata[tid] + frsum_sdata[tid + offset];
        }
        __syncthreads();
    }
    for (uint offset = 16; offset > 0; offset >>= 1) {
        if (tid < offset) frsum_sdata[tid] += frsum_sdata[tid + offset];
        __syncwarp();
    }

    if (tid == 0) output[blockIdx.x] = frsum_sdata[0];
}

__global__ void sum_by_dim_kernel(fr_t* in, fr_t* out, uint N, uint dim) 
{
    const uint gid = GET_GLOBAL_ID();
    if (gid >= N) return;
    fr_t sum;   sum.zero();
    for(uint i = 0; i < dim; i++)  sum += in[gid * dim + i];
    for(uint i = 0; i < dim; i++)  out[gid * dim + i] = sum;
  
}



// 辅助函数定义------------------------------------------------------------------------------------------------------------
void Fr_partial_me(uint N, FrTensor& t, fr_t *x, uint id, uint cur_dim, uint window_size)
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

        Fr_partial_eval_kernel<<<(out_size + 256 -1) / 256, 256>>>(t.gpu_data, x, dim, other_dims, cur_dim_in, cur_dim_out, id, window_size, gap);
        cudaDeviceSynchronize();
        id += 1;
        gap = gap << 1;
        cur_dim_in = cur_dim_out;
    }
    return;
}

void pad_int(int **int_data, uint N, uint pad_N, uint pad_size) {
    if(N == pad_N) return;
    int *int_data_ptr = *int_data; 
    int *new_data;
    cudaMalloc((void **)&new_data, pad_size * sizeof(int));
    pad_int_kernel<<<(pad_size + 256 - 1) / 256, 256>>>(int_data_ptr, new_data, N, pad_N, pad_size);
    cudaDeviceSynchronize();

    *int_data = new_data;
    cudaFree(int_data_ptr);
}

// FrTensor 类方法定义--------------------------------------------------------------------------------------------------------
FrTensor& FrTensor::operator*=(fr_t *x)
{
    Fr_broadcast_mul<<<(size + 256 -1) / 256, 256>>>(gpu_data, x, gpu_data, size);
    cudaDeviceSynchronize();
    return *this;
}

FrTensor& FrTensor::operator*=(Fr &x)
{
    fr_t *d_x;
    cudaMalloc(&d_x, sizeof(fr_t));
    cudaMemcpy(d_x, &x, sizeof(fr_t), cudaMemcpyHostToDevice);
    Fr_broadcast_mul<<<(size + 256 -1) / 256, 256>>>(gpu_data, d_x, gpu_data, size);
    cudaDeviceSynchronize();

    cudaFree(d_x);
    return *this;
}

FrTensor& FrTensor::operator*=(const uint x)
{
    Fr host_x = Fr(long(x));
    fr_t *d_x;
    cudaMalloc(&d_x, sizeof(fr_t));
    cudaMemcpy(d_x, &host_x, sizeof(fr_t), cudaMemcpyHostToDevice);
    Fr_broadcast_mul<<<(size + 256 -1) / 256, 256>>>(gpu_data, d_x, gpu_data, size);
    cudaDeviceSynchronize();

    cudaFree(d_x);
    return *this;
}

FrTensor::~FrTensor()
{
    if (gpu_data != nullptr) {
        cudaFree(gpu_data);
        gpu_data = nullptr;
    }
}

void FrTensor::set_size(uint sz)
{ 
    assert(gpu_data != nullptr);
    size = sz;
}

FrTensor& FrTensor::operator+=(const FrTensor& t)
{
    if (size != t.size) throw std::runtime_error("Incompatible dimensions");
    Fr_elementwise_add<<<(size + 256 - 1) / 256 , 256>>>(gpu_data, t.gpu_data, gpu_data, size);
    cudaDeviceSynchronize();
    return *this;
}

FrTensor& FrTensor::operator+=(const int x)
{
    Fr host_x = Fr(long(x));
    fr_t *d_x;
    cudaMalloc(&d_x, sizeof(fr_t));
    cudaMemcpy(d_x, &host_x, sizeof(fr_t), cudaMemcpyHostToDevice);
    Fr_broadcast_add<<<(size + 256 -1) / 256, 256>>>(gpu_data, d_x, gpu_data, size);
    cudaDeviceSynchronize();

    cudaFree(d_x);
    return *this;
}

FrTensor& FrTensor::operator-=(const FrTensor& t)
{
    if (size != t.size) throw std::runtime_error("Incompatible dimensions");
    Fr_elementwise_sub<<<(size + 256 - 1) / 256 , 256>>>(gpu_data, t.gpu_data, gpu_data, size);
    cudaDeviceSynchronize();
    return *this;
}

void FrTensor::add_with_size(uint sz, const FrTensor& t)
{
    Fr_elementwise_add<<<(sz + 256 - 1) / 256 , 256>>>(gpu_data, t.gpu_data, gpu_data, sz);
    cudaDeviceSynchronize();
}

void FrTensor::mul_with_size(uint sz, const FrTensor& t)
{
    Fr_elementwise_mul<<<(sz + 256 - 1) / 256 , 256>>>(gpu_data, t.gpu_data, gpu_data, sz);
    cudaDeviceSynchronize();
}

void FrTensor::mul_with_size(uint sz, fr_t *x)
{
    Fr_broadcast_mul<<<(sz + 256 - 1) / 256 , 256>>>(gpu_data, x, gpu_data, sz);
    cudaDeviceSynchronize();
}

void FrTensor::mul_with_size(uint sz, Fr &x)
{
    fr_t *d_x;
    cudaMalloc(&d_x, sizeof(fr_t));
    cudaMemcpy(d_x, &x, sizeof(fr_t), cudaMemcpyHostToDevice);
    Fr_broadcast_mul<<<(sz + 256 - 1) / 256 , 256>>>(gpu_data, d_x, gpu_data, sz);
    cudaDeviceSynchronize();

    cudaFree(d_x);
}

FrTensor& FrTensor::operator*=(const FrTensor& t)
{
    if (size != t.size) throw std::runtime_error("Incompatible dimensions");
    Fr_elementwise_mul<<<(size + 256 - 1) / 256 , 256>>>(gpu_data, t.gpu_data, gpu_data, size);
    cudaDeviceSynchronize();
    return *this;
}

FrTensor::FrTensor(const FrTensor& t): size(t.size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(fr_t) * size);
    cudaMemcpy(gpu_data, t.gpu_data, sizeof(fr_t) * size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}

FrTensor& FrTensor::operator=(const FrTensor& t)
{
    uint small_size = std::min(size, t.size);
    cudaMemcpy(gpu_data, t.gpu_data, sizeof(fr_t) * small_size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    return *this;
}

void FrTensor::partial_eval(uint N, fr_t *x, uint c, uint id, uint gap)
{
    Fr_partial_me(N, *this, x, id, c, gap);
    
    if(gap == 1 && (N != c)){
        uint thread = (N / c) > 256 ? 256 : (N / c);
        Memory_alignment<<<((N / c) + 256 -1) / 256, thread>>>(gpu_data, c);
    }
    else if(gap != 1){
        for(int i = 1; i < (N / gap / c); i++){
            cudaMemcpy(gpu_data + i * gap, gpu_data + i * c * gap , sizeof(fr_t) * gap, cudaMemcpyDeviceToDevice);
        }
    }
    cudaDeviceSynchronize();
}

Fr FrTensor::operator()(const std::vector<Fr>& u) 
{
    uint log_dim = u.size();

    fr_t *d_u;
    cudaMalloc(&d_u, sizeof(fr_t) * log_dim);
    cudaMemcpy(d_u, u.data(), sizeof(fr_t) * log_dim, cudaMemcpyHostToDevice);

    if (size < (1 << log_dim)){
        Fr_partial_me(size, *this, d_u, Log2(11008), size / 11008, 11008);
        cudaDeviceSynchronize();
        Fr_partial_me(11008, *this, d_u, 0, 11008, 1);
        cudaDeviceSynchronize();
    } 
    else{
        Fr_partial_me(size, *this, d_u, 0, size, 1);
        cudaDeviceSynchronize();
    }

    Fr ret;
    cudaMemcpy(&ret, gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);
    cudaFree(d_u);
    return ret;
}

FrTensor::FrTensor(uint size): size(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(fr_t) * size);
}

FrTensor::FrTensor(uint size, int* int_data): size(size), gpu_data(nullptr)
{
    cudaMalloc((void **)&gpu_data, sizeof(fr_t) * size);
    int_to_fr<<<(size + 256 - 1) / 256, 256>>>(int_data, gpu_data, size);
    cudaDeviceSynchronize();
}

void FrTensor::get_data(uint sz, int *data)
{ 
    size = sz;
    int_to_fr<<<(size + 256 - 1) / 256, 256>>>(data, gpu_data, size);
    cudaDeviceSynchronize();
}

void FrTensor::get_data(uint sz, uint *data)
{ 
    size = sz;
    uint_to_fr<<<(size + 256 - 1) / 256, 256>>>(data, gpu_data, size);
    cudaDeviceSynchronize();
}

void FrTensor::from()
{
    from_kernel<<<(size + 256 - 1) / 256, 256>>>(gpu_data, size);
    cudaDeviceSynchronize();
}

void FrTensor::to()
{
    to_kernel<<<(size + 256 - 1) / 256, 256>>>(gpu_data, size);
    cudaDeviceSynchronize();
}

void FrTensor::inverse()
{
    inverse_kernel<<<(size + 256 - 1) / 256 , 256>>>(gpu_data, size);
    cudaDeviceSynchronize();
}

Fr FrTensor::sum(uint sz)
{
    fr_t *ptr_input, *ptr_output;
    uint curSize = sz;
    uint gridSize = (curSize + 256 - 1) / 256;
    cudaMalloc((void**)&ptr_output, gridSize * sizeof(fr_t));
    fr_t *free_pointer = ptr_output;
    ptr_input = gpu_data;

    while(curSize > 1) {
        Fr_sum_reduction<<<gridSize, 256, 2 * sizeof(fr_t) * 256>>>(ptr_input, ptr_output, curSize);
        cudaDeviceSynchronize(); 
        
        fr_t *temp = ptr_input;
        ptr_input = ptr_output;
        ptr_output = temp;
        
        curSize = gridSize;
        gridSize = (curSize + 256 - 1) / 256;
    }
    
    Fr out;
    cudaMemcpy(&out, ptr_input, sizeof(Fr), cudaMemcpyDeviceToHost);
    cudaFree(free_pointer);
    return out;
}

FrTensor FrTensor::sum(uint sz, uint N)
{
    FrTensor out(sz);
    assert(sz % N == 0);
    sum_by_dim_kernel<<<((sz / N) + 256 - 1) / 256 , 256>>>(gpu_data, out.gpu_data, size / N, N); 
    cudaDeviceSynchronize();
    return out;
}


Fr FrTensor::operator()(uint idx) const
{
    Fr out;
    cudaMemcpy(&out, gpu_data + idx, sizeof(Fr), cudaMemcpyDeviceToHost);
    return out;
}

void FrTensor::pad(FrTensor &tmp, const uint N, const uint pad_N, const uint pad_size, const Fr &pad_val){
    if(N == pad_N) return;
    fr_t *d_pad_val;
    cudaMalloc(&d_pad_val, sizeof(fr_t));  cudaMemcpy(d_pad_val, &pad_val, sizeof(fr_t), cudaMemcpyHostToDevice);
    pad_fr_kernel<<<(pad_size + 256 - 1) / 256, 256>>>(gpu_data, tmp.gpu_data, N, pad_N, pad_size, d_pad_val); 
    cudaDeviceSynchronize();
    cudaMemcpy(gpu_data, tmp.gpu_data, pad_size * sizeof(fr_t), cudaMemcpyDeviceToDevice);
    size = pad_size;
}

