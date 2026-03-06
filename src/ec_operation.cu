#include "ec_operation.cuh"

// ---------------- kernel implementations ----------------
__global__ void check_G_equal(const jacob_t *a, const jacob_t *b, int n)
{
    int idx = GET_GLOBAL_ID();
    if (idx >= n) return;
    jacob_t res;
    res = b[idx];
    res.cneg(1);
    res.add(a[idx]);
    assert(res.is_inf());
}

__global__ void naive_msm(jacob_t *P, affine_t *out, size_t N, size_t W)
{
    uint global_id = GET_GLOBAL_ID();
    if (global_id >= N) return;
    jacob_t tmp = P[global_id];
    jacob_t res;
    res.inf();
    
    for (int w = 0; w < W; w++) {
        out[w * N + global_id] = res;
        res.add(tmp);
    }
}

// ---------------- host implementations ----------------
vector<bn128> precompute_generators(uint N, uint W, affine_t *out)
{
    vector<bn128> generators(N);
    for (uint i = 0; i < N; i++) {
        generators[i] = bn128::random_element();
    }

    jacob_t *gpu_points;
    cudaMalloc((void **)&gpu_points, sizeof(jacob_t) * N);
    cudaMemcpy(gpu_points, generators.data(), sizeof(jacob_t) * N, cudaMemcpyHostToDevice);

    CUDA_TIMER_START(pre_points);
    naive_msm<<<(N + 255) / 256, 256>>>(gpu_points, out, N, W);
    CUDA_DEBUG;
    CUDA_TIMER_STOP(pre_points);

    cudaFree(gpu_points);
    return generators;
}

vector<Fr> random_vec(uint len)
{
    vector<Fr> out(len);
    for (uint i = 0; i < len; i++)
        out[i] = Fr::random_element();
    return out;
}

void generate_random_eval_points(size_t data_size, vector<Fr>& eval_point) {
    size_t log_size = Log2(data_size);
    
    // 只有当大小改变时才重新分配
    if (eval_point.size() != log_size) {
        eval_point.resize(log_size);
    }
    
    #pragma omp parallel for
    for(int i = 0; i < log_size; i++){
        eval_point[i] = Fr::random_element();
    }
}


void ReassembleVectors(std::vector<Fr>& first,
                    std::vector<Fr>& second,
                    size_t first_size,
                    size_t same_size,            //这两个size都是求过log后的
                    size_t last_size) {

    // 调整result的大小
    vector<Fr> result1(second.size() + first_size);
    vector<Fr> result2(last_size + second.size());

    // 复制second的前部分
    std::copy(second.begin(), 
              second.begin() + same_size, 
              result1.begin());
    
    // 接着复制first的中间部分
    std::copy(first.begin() + last_size, 
              first.begin() + first_size + last_size, 
              result1.begin() + same_size);

    //最后赋值second的剩余部分
    std::copy(second.begin() + same_size, 
              second.end(),
              result1.begin() + first_size + same_size);

    // -------------------------------第一个vector done---------------------------------
    
    // 复制second的前部分
    std::copy(second.begin(), 
              second.begin() + same_size, 
              result2.begin());
    
    // 接着复制first的前部分
    std::copy(first.begin(), 
              first.begin() + last_size, 
              result2.begin() + same_size);

    //最后赋值second的剩余部分
    std::copy(second.begin() + same_size, 
              second.end(),
              result2.begin() + same_size + last_size);
    // -------------------------------第二个vector done---------------------------------

    second.resize(result2.size());
    std::copy(result2.begin(), result2.end(), second.begin());

    first.resize(result1.size());
    std::copy(result1.begin(), result1.end(), first.begin());
}