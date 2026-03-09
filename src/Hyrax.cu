#include "Hyrax.cuh"

// 设备函数定义-------------------------------------------------------------------------------------------------
__device__ __forceinline__ fp_t shfl_down_Fp(
    const fp_t &a,
    unsigned int offset,
    int width,
    unsigned mask)
{
    fp_t r;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        r[i] = __shfl_down_sync(mask, a[i], offset, width);
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

__device__ __forceinline__ bucket_t shfl_down_xyzz(
    const bucket_t &p,
    unsigned int offset,
    int width,
    unsigned mask)
{
    fp_t X = shfl_down_Fp(p.getX(), offset, width, mask);
    fp_t Y = shfl_down_Fp(p.getY(), offset, width, mask);
    fp_t ZZZ = shfl_down_Fp(p.getZZZ(), offset, width, mask);
    fp_t ZZ = shfl_down_Fp(p.getZZ(), offset, width, mask);
    bucket_t r(X, Y, ZZZ, ZZ);
    return r;
}

// 内核函数定义-------------------------------------------------------------------------------------------------------------
__global__ void int_windows_sum(int* scalars, affine_t *point, jacob_t *out, uint size, uint N, uint npoints)
{
    uint global_warp_id = GET_GLOBAL_WARP_ID();
    uint lane_id = GET_LANE_ID();
    unsigned int mask = (1 << 8) - 1;

    for(int w = 3; w >=0; w--){
        bucket_t sum; sum.inf();
        for(int i = lane_id; i < N; i += 32){
            bool is_neg = scalars[global_warp_id * N + i] < 0 ? 1 : 0;
            uint s = abs(scalars[global_warp_id * N + i]);
            uint index = reinterpret_cast<uint8_t*>(&s)[w];
            affine_t tmp = point[index * npoints + i];
            tmp.cneg(is_neg);
            sum.add(tmp);
        }
        sum.to_jacobian();
        out[global_warp_id * 128 + w * 32 + lane_id] = *reinterpret_cast<jacob_t*>(&sum);
    }
}

__global__ void int_windows_reduce(jacob_t *in, jacob_t *out)
{
    __shared__ jacob_t sm[128];
    uint lane_id = threadIdx.x % 4;
    uint global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 4; 
    jacob_t sum; sum.inf();
    for(int i = 0; i < 32; i++){
        sum.add(in[global_warp_id * 128 + lane_id * 32 + i]);
    }

    for(int i = 0; i < lane_id * 8; i++)  sum.dbl();
    sm[threadIdx.x] = sum;
    __syncthreads();
   
    if(lane_id == 0){
        sum.add(sm[threadIdx.x + 1]);
        sum.add(sm[threadIdx.x + 2]);
        sum.add(sm[threadIdx.x + 3]);
        out[global_warp_id] = sum;
    }
}

__global__ void single_windows_sum(fr_t* scalars, affine_t *point, jacob_t *out, uint N, uint npoints)
{
    uint lane_id = GET_LANE_ID();
    uint block_id = blockIdx.x;
    uint grid_size = gridDim.x;
    unsigned int mask = (1 << 8) - 1;
    jacob_t sum; sum.inf();
    for(int i = block_id; i < N; i += grid_size){
        fr_t s = scalars[i];
        s.from();
        uint index = (s[lane_id / 4] >> ((lane_id % 4) * 8)) & mask;
        sum.add(point[index * npoints + i]);
        __syncwarp();
    }

    out[block_id * 32 + lane_id] = sum;
}

__global__ void single_windows_reduce(jacob_t *in, jacob_t *out, uint num)
{
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

__global__ void many_windows_sum(fr_t* scalars, affine_t *point, jacob_t *out, uint N, uint npoints)
{
    uint lane_id = GET_LANE_ID();
    uint warp_id = GET_WARP_ID();
    unsigned int mask = (1 << 8) - 1;
    bucket_t sum; sum.inf();

    uint block_id = blockIdx.x;
    uint stride = blockDim.x / 32;

    for(int i = warp_id; i < N; i += stride){
        uint index = reinterpret_cast<uint8_t*>(&scalars[block_id * N + i])[lane_id];
        sum.add(point[index * npoints + i]);
    }
    sum.to_jacobian();
    out[block_id * blockDim.x + threadIdx.x] = *reinterpret_cast<jacob_t*>(&sum);
}

__global__ void many_windows_reduce(jacob_t *in, jacob_t *out, uint num)
{
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

__global__ void affine_to_jacob(jacob_t *d, affine_t *h, uint size)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    d[tid] = h[tid + size];
}

// 辅助函数定义---------------------------------------------------------------------------------------------------------
void single_commit(fr_t *d, affine_t *g, jacob_t *com, uint N, uint npoints)    //open  commit d需要
{ 
    jacob_t *tmp_out;
    cudaMalloc(&tmp_out, sizeof(jacob_t) * 32 * 32);
    single_windows_sum<<<32, 32>>>(d, g, tmp_out, N, npoints);
    cudaDeviceSynchronize();
    single_windows_reduce<<<1, 32>>>(tmp_out, com, 32);
    cudaDeviceSynchronize();
    cudaFree(tmp_out);
}

// 类方法定义-------------------------------------------------------------------------------------------------------------
Hyrax_proof::Hyrax_proof(Fr result_, size_t z_len, bn128 commit_d_, bn128 commit_ad_)
    : result(result_), z_size(z_len),
      commit_d(commit_d_), commit_ad(commit_ad_)
{
    z = new Fr[z_size];
}

Hyrax_proof::~Hyrax_proof()
{
    delete[] z;
}

Hyrax_proof::Hyrax_proof(const Hyrax_proof& other)
        : z_size(other.z_size), result(other.result),
          commit_d(other.commit_d), commit_ad(other.commit_ad) {
        z = new Fr[z_size];
        for (size_t i = 0; i < z_size; i++) {
            z[i] = other.z[i];
        }
    }

Hyrax_proof& Hyrax_proof::operator=(const Hyrax_proof& other) {
        if (this != &other) {  // 防止自赋值
            delete[] z;
            
            // 2. 复制基本类型成员
            z_size = other.z_size;
            result = other.result;
            commit_d = other.commit_d;
            commit_ad = other.commit_ad;
            
            // 3. 分配新内存并复制数据
            z = new Fr[z_size];
            for (size_t i = 0; i < z_size; i++) {
                z[i] = other.z[i];
            }
        }
        return *this;
    }

Hyrax::Hyrax(uint layer_num_, uint npoints_, affine_t *g, bn128 &G_)
    : layer_num(layer_num_), npoints(npoints_ * layer_num_), g_affine(g), G(G_)
{
}

Hyrax::~Hyrax()
{
}

Hyrax_proof Hyrax::open(FrTensor &tensor, const vector<Fr> eval_point_, Fr c, uint size, uint N)
{
    if(size / N / layer_num < layer_num)  throw std::runtime_error("Size too small: size < N * layer_num * layer_num");

    uint pad_N = 1 << Log2(N);
    uint new_N = pad_N * layer_num;

    assert(eval_point_.size() == Log2(size));
    fr_t *eval_point;
    cudaMalloc(&eval_point, sizeof(fr_t) * eval_point_.size());
    cudaMemcpy(eval_point, eval_point_.data(), sizeof(fr_t) * eval_point_.size(), cudaMemcpyHostToDevice);
    
    fr_t *c_dev; 
    cudaMalloc(&c_dev, sizeof(fr_t)); 
    cudaMemcpy(c_dev, &c, sizeof(fr_t), cudaMemcpyHostToDevice);

    Fr *host_d = new Fr[new_N];

    #pragma omp parallel for
    for(uint i = 0; i < new_N; i++) host_d[i] = Fr::random_element();
    FrTensor d(new_N);
    cudaMemcpy(d.gpu_data, host_d, sizeof(fr_t) * new_N, cudaMemcpyHostToDevice);
    
    jacob_t *commit_d;
    cudaMalloc(&commit_d, sizeof(jacob_t));

    single_commit(d.gpu_data, g_affine, commit_d, new_N, npoints);

    FrTensor x(new_N);

    
    tensor.partial_eval(size, eval_point, (size / layer_num) / (N * layer_num), Log2(N * layer_num), N * layer_num);
    //这里不能用new_N,因为tensor并没有pad   这里两行很关键
    tensor.partial_eval((N * layer_num) * layer_num, eval_point, layer_num, Log2(size / layer_num), N * layer_num);

    x = tensor;
    if(pad_N != N) x.pad(tensor, N, pad_N, new_N, Fr::zero());
    
    Fr result;
    
    tensor.partial_eval(new_N, eval_point, new_N, 0, 1);  //这里就可以用new_N了，因为在pad x的时候tensor作为缓存也pad了
   
    cudaMemcpy(&result, tensor.gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);

    FrTensor z(new_N);
    z = x;
    z *= c_dev;
    z += d;

    d.partial_eval(d.size, eval_point, new_N, 0, 1);
    
    Fr sum_ad;
    cudaMemcpy(&sum_ad, d.gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);
    bn128 commit_ad;
    commit_ad = sum_ad * G;
    
    bn128 host_com_d;
    cudaMemcpy(&host_com_d, commit_d, sizeof(bn128), cudaMemcpyDeviceToHost);

    Hyrax_proof proof(result, new_N, host_com_d, commit_ad);

    cudaMemcpy(proof.z, z.gpu_data, sizeof(fr_t) * new_N, cudaMemcpyDeviceToHost);
   
    cudaFree(c_dev);
    cudaFree(eval_point);
    cudaFree(commit_d);
    delete [] host_d;
    return proof;
}

void Hyrax::verify(Hyrax_proof &proof, jacob_t *commitment, Fr c, uint size, uint N)
{ 

}

jacob_t* Hyrax::commit(int *tensor, uint size, uint N)
{
    if(size / N / layer_num < layer_num)  throw std::runtime_error("Size too small: size < N * layer_num * layer_num");
    if(N != 1 << Log2(N)) throw std::runtime_error("N must be a power of 2");
    N = N * layer_num;
    jacob_t *commitment;
    cudaMalloc(&commitment, sizeof(jacob_t) * (size / N));

    jacob_t *tmp_out;
    cudaMalloc(&tmp_out, sizeof(jacob_t) * (size / N) * 32 * 4);

    uint thread = (size / N) * 32;
    int_windows_sum<<<(thread + 128 - 1) / 128, 128>>>(tensor, g_affine, tmp_out, size, N, npoints);
    cudaDeviceSynchronize();
    
    thread = (size / N) * 4;
    int_windows_reduce<<<(thread + 128 - 1) / 128, 128>>>(tmp_out, commitment);
    cudaDeviceSynchronize();
  
    cudaFree(tmp_out);
    return commitment;
}

jacob_t* Hyrax::commit(FrTensor &tensor, uint size, uint N)
{ 
    if(size / N / layer_num < layer_num)  throw std::runtime_error("Size too small: size < N * layer_num * layer_num");
    if(N !=1 << Log2(N)) throw std::runtime_error("N must be a power of 2");

    N = N * layer_num;
    jacob_t *commitment;
    cudaMalloc(&commitment, sizeof(jacob_t) * (size / N));
    uint threads_per_block = 128;
    jacob_t *tmp_out;
    cudaMalloc(&tmp_out, sizeof(jacob_t) * (size / N) * threads_per_block);
    
    tensor.from();
    
    many_windows_sum<<<(size / N), threads_per_block>>>(tensor.gpu_data, g_affine, tmp_out, N, npoints);
    cudaDeviceSynchronize();
    
    uint blocknum = 32 * (size / N) / threads_per_block;
    many_windows_reduce<<<blocknum, threads_per_block>>>(tmp_out, commitment, threads_per_block / 32);
    cudaDeviceSynchronize();

    tensor.to();
    cudaFree(tmp_out);
    return commitment;
}