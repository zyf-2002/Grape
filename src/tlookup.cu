#include "tlookup.cuh"


// 内核函数定义-----------------------------------------------------------------------------------------------------------
__global__ void get_histogram_kernel(const fr_t* value, uint *hist, uint size, fr_t *low){
    uint gid = GET_GLOBAL_ID();
    if(gid < size){
        fr_t tmp;
        tmp = value[gid] - low[0];
        tmp.from();
        uint index = tmp[0];
        atomicAdd(&hist[index], 1U);
    }
}

__global__ void tlookuprange_init_kernel(fr_t* table_ptr, fr_t* inv_table_ptr, const fr_t *low, 
                                                const uint table_size, const fr_t *beta)
{
    uint gid = GET_GLOBAL_ID();
    if (gid < table_size)
    {
        fr_t tmp;
        tmp.set_uint(gid);
        tmp.to();  
        tmp += low[0];
        table_ptr[gid] = tmp;
        tmp += beta[0];
        tmp = tmp.reciprocal();
        inv_table_ptr[gid] = tmp;
    }
}

__global__ void tLookupRangeMapping_init_kernel(fr_t* table_ptr, fr_t* inv_table_ptr, fr_t *map_table_ptr, const fr_t *low, 
                                                const uint table_size, const fr_t *r, const fr_t *beta)
{
    uint gid = GET_GLOBAL_ID();
    if (gid < table_size)
    { 
        fr_t tmp;
        tmp.set_uint(gid);
        tmp.to();  
        tmp += low[0];
        tmp += map_table_ptr[gid] * r[0];
        table_ptr[gid] = tmp;
        tmp += beta[0];
        tmp = tmp.reciprocal();
        inv_table_ptr[gid] = tmp;
    }

}

__global__ void tlookup_inv_kernel(fr_t* value, fr_t* value_inv, fr_t *beta, uint size)
{ 
    uint gid = GET_GLOBAL_ID();
    if (gid < size)
    {
        fr_t tmp;
        tmp = value[gid];
        tmp += beta[0];
        value_inv[gid] = tmp.reciprocal();
    }
}

__global__ void tLookup_poly_kernel(const fr_t* A_data, const fr_t* S_data, fr_t *beta, fr_t* out0, fr_t* out1, fr_t* out2, uint N_out)
{
    const uint tid = GET_GLOBAL_ID();
    if (tid < N_out)
    {
        fr_t c00 = A_data[tid];
        fr_t c01 = A_data[tid + N_out] - A_data[tid];
        fr_t c10 = S_data[tid] + beta[0];
        fr_t c11 = S_data[tid + N_out] - S_data[tid];

        out0[tid] = c00 * c10;
        out1[tid] = c01 * c10 + c00 * c11;
        out2[tid] = c01 * c11;
    }
}

__global__ void tLookup_poly_sum_kernel(const fr_t* A_data, fr_t* out0, fr_t* out1, uint N_out)
{
    const uint tid = GET_GLOBAL_ID();
    if (tid < N_out)
    {
        out0[tid] = A_data[tid];
        out1[tid] = A_data[tid + N_out] - A_data[tid];
    }
}

__global__ void tLookup_phase1_reduce_kernel(const fr_t* A_data, const fr_t* S_data, fr_t* new_A_data, fr_t* new_S_data, fr_t *v, uint N_out)
{
    const uint tid = GET_GLOBAL_ID();
    if (tid < N_out)
    {
        new_A_data[tid] = A_data[tid] + v[0] * (A_data[tid + N_out] - A_data[tid]);
        new_S_data[tid] = S_data[tid] + v[0] * (S_data[tid + N_out] - S_data[tid]);
    }
}

__global__ void tLookup_phase2_reduce_kernel(const fr_t* A_data, const fr_t* S_data, const fr_t* B_data, const fr_t* T_data, const fr_t* m_data,
    fr_t* new_A_data, fr_t* new_S_data, fr_t* new_B_data, fr_t* new_T_data, fr_t* new_m_data,
    fr_t *v, uint N_out)
{
    const uint tid = GET_GLOBAL_ID();
    if (tid < N_out)
    {
        new_A_data[tid] = A_data[tid] + v[0] * (A_data[tid + N_out] - A_data[tid]);  
        new_S_data[tid] = S_data[tid] + v[0] * (S_data[tid + N_out] - S_data[tid]);
        new_B_data[tid] = B_data[tid] + v[0] * (B_data[tid + N_out] - B_data[tid]);
        new_T_data[tid] = T_data[tid] + v[0] * (T_data[tid + N_out] - T_data[tid]);
        new_m_data[tid] = m_data[tid] + v[0] * (m_data[tid + N_out] - m_data[tid]);
    }
}

Timer timer;
// 辅助函数定义--------------------------------------------------------------------------------------------
Polynomial tLookup_phase1_step_poly(const Fr& claim, const FrTensor& A, const FrTensor& S, 
    const Fr& alpha, const Fr& beta, const Fr& C, const vector<Fr>& u)
{
    if (A.size != S.size) throw std::runtime_error("A.size != S.size");
    uint D = A.size;

    //CUDA_TIMER_START(mallo);
    FrTensor temp0(D >> 1), temp1(D >> 1), temp2(D >> 1);
    //CUDA_TIMER_STOP(mallo);
    fr_t *d_beta;
    cudaMalloc(&d_beta, sizeof(fr_t));  cudaMemcpy(d_beta, &beta, sizeof(fr_t), cudaMemcpyHostToDevice);
    //CUDA_TIMER_START(tLookup_poly_kernel);
    tLookup_poly_kernel<<<((D >> 1)+256-1)/256,256>>>(
        A.gpu_data, S.gpu_data, d_beta, temp0.gpu_data, temp1.gpu_data, temp2.gpu_data, D >> 1);
    cudaDeviceSynchronize();
    //CUDA_TIMER_STOP(tLookup_poly_kernel);

    vector<Fr> u_(u.begin(), u.end() - 1);
    //CUDA_TIMER_START(eval);
    Polynomial p0 ({temp0(u_), temp1(u_), temp2(u_)}); 
    //CUDA_TIMER_STOP(eval);
    p0 = p0 * alpha;
    p0 *= Polynomial::eq(u.back());

    
    tLookup_poly_sum_kernel<<<((D >> 1)+256-1)/256,256>>>(
        A.gpu_data, temp0.gpu_data, temp1.gpu_data, D >> 1);
    cudaDeviceSynchronize();

    Polynomial p1 ({temp0.sum(D >> 1), temp1.sum(D >> 1)});
    
    cudaFree(d_beta);
    Fr two_inv = Fr(2).invert();
    return p0 + p1 + C * two_inv;
}

Polynomial tLookup_phase2_step_poly(const FrTensor& A, const FrTensor& S, const FrTensor& B, const FrTensor& T, const FrTensor& m,
    const Fr& alpha_, const Fr& beta, const Fr& inv_size_ratio, const Fr& alpha_sq,
    const vector<Fr>& u)
{
    uint N = m.size;
    uint N_out = N >> 1;
    vector<Fr> u_(u.begin(), u.end() - 1);
    FrTensor temp0(N_out), temp1(N_out), temp2(N_out);

    fr_t *d_beta;
    cudaMalloc(&d_beta, sizeof(fr_t));   cudaMemcpy(d_beta, &beta, sizeof(fr_t), cudaMemcpyHostToDevice);
    tLookup_poly_kernel<<<(N_out+256-1)/256,256>>>(
        A.gpu_data, S.gpu_data, d_beta, temp0.gpu_data, temp1.gpu_data, temp2.gpu_data, N_out);
    cudaDeviceSynchronize();
    Polynomial p0 ({temp0(u_), temp1(u_), temp2(u_)});
    p0 = p0 * alpha_;
        
    Fr coef = inv_size_ratio * alpha_sq;
    tLookup_poly_kernel<<<(N_out+256-1)/256,256>>>(
        B.gpu_data, T.gpu_data, d_beta, temp0.gpu_data, temp1.gpu_data, temp2.gpu_data, N_out);
    cudaDeviceSynchronize();
    p0 += {{temp0(u_) * coef, temp1(u_) * coef, temp2(u_) * coef}};

    tLookup_poly_sum_kernel<<<(N_out+256-1)/256,256>>>(
        A.gpu_data, temp0.gpu_data, temp1.gpu_data, N_out);
    cudaDeviceSynchronize();
    Polynomial p1 ({temp0.sum(N_out), temp1.sum(N_out)});

    cudaMemset(d_beta, 0, sizeof(fr_t));
    tLookup_poly_kernel<<<(N_out+256-1)/256,256>>>(
        m.gpu_data, B.gpu_data, d_beta, temp0.gpu_data, temp1.gpu_data, temp2.gpu_data, N_out
    );
    cudaDeviceSynchronize();
    Polynomial p2 ({temp0.sum(N_out), temp1.sum(N_out), temp2.sum(N_out)});

    cudaFree(d_beta);
    return Polynomial::eq(u.back()) * p0 + p1 - p2 * inv_size_ratio; 
}

Fr tLookup_phase2(const Fr& claim, const FrTensor& A, const FrTensor& S, const FrTensor& B, const FrTensor& T, const FrTensor& m,
    const Fr& alpha_, const Fr& beta, const Fr& inv_size_ratio, const Fr& alpha_sq, const vector<Fr>& u, const vector<Fr>& v2)
{
    if (!v2.size()) return S(0);
    
    //CPU_TIMER_START(step2);
    auto p = tLookup_phase2_step_poly(A, S, B, T, m, alpha_, beta, inv_size_ratio, alpha_sq, u);
    //cudaDeviceSynchronize();
    //CPU_TIMER_STOP(step2);

    FrTensor new_A(A.size >> 1), new_S(S.size >> 1), new_B(B.size >> 1), new_T(T.size >> 1), new_m(m.size >> 1);
    if (claim != p(0) + p(1)) throw std::runtime_error("tLookup_phase2: claim != p(0) + p(1)");

    fr_t *d_v2; cudaMalloc(&d_v2, sizeof(fr_t));  cudaMemcpy(d_v2, &v2.back(), sizeof(fr_t), cudaMemcpyHostToDevice);
    tLookup_phase2_reduce_kernel<<<((A.size >> 1)+256-1)/256,256>>>(
        A.gpu_data, S.gpu_data, B.gpu_data, T.gpu_data, m.gpu_data,
        new_A.gpu_data, new_S.gpu_data, new_B.gpu_data, new_T.gpu_data, new_m.gpu_data,
        d_v2, A.size >> 1
    );
    cudaDeviceSynchronize();
    cudaFree(d_v2);
    return tLookup_phase2(p(v2.back()), new_A, new_S, new_B, new_T, new_m, alpha_ * Polynomial::eq(u.back(), v2.back()), beta, inv_size_ratio, alpha_sq * Polynomial::eq(u.back(), v2.back()), {u.begin(), u.end() - 1}, {v2.begin(), v2.end() - 1});
}

Fr tLookup_phase1(const Fr& claim, FrTensor& A, FrTensor& S, const FrTensor& B, const FrTensor& T, FrTensor& m,
    const Fr& alpha, const Fr& beta, const Fr& C, const Fr& inv_size_ratio, const Fr& alpha_sq, 
    const vector<Fr>& u, const vector<Fr>& v1, const vector<Fr>& v2)
{
    if (!v1.size())
    {
        return tLookup_phase2(claim, A, S, B, T, m, alpha, beta, inv_size_ratio, alpha_sq, u, v2);
    }
    else{

        //CPU_TIMER_START(step1);
        auto p = tLookup_phase1_step_poly(claim, A, S, alpha, beta, C, u);
        //CPU_TIMER_STOP(step1);
        //CUDA_DEBUG;
        
        if (claim != p(0) + p(1)) throw std::runtime_error("tLookup_phase1: claim != p(0) + p(1)");
       
        fr_t *d_v;  cudaMalloc(&d_v, sizeof(fr_t));  cudaMemcpy(d_v, &v1.back(), sizeof(fr_t), cudaMemcpyHostToDevice);
        
        A.partial_eval(A.size, d_v, 2, 0, A.size >> 1);
        A.set_size(A.size >> 1);
        S.partial_eval(S.size, d_v, 2, 0, S.size >> 1);
        S.set_size(S.size >> 1);
        cudaFree(d_v);
        cudaDeviceSynchronize();
        Fr two_inv = Fr(2).invert();
        return tLookup_phase1(p(v1.back()), A, S, B, T, m, alpha * Polynomial::eq(u.back(), v1.back()), beta, C * two_inv, inv_size_ratio, alpha_sq, {u.begin(), u.end() - 1}, {v1.begin(), v1.end() - 1}, v2);
    }
}

// 类方法定义----------------------------------------------------------------------------------------------------------------
tLookup::tLookup(const uint size): T(size), B(size), m(size), alpha(Fr::random_element()), beta(Fr(size))
{
}


tLookupRange::tLookupRange(int low, uint len) : low(low), tLookup(1 << Log2(len))
{
    fr_t *d_beta;
    cudaMalloc(&d_beta, sizeof(fr_t));
    cudaMemcpy(d_beta, &beta, sizeof(fr_t), cudaMemcpyHostToDevice);
    Fr low_bound = Fr(low);
    fr_t *d_low_bound;
    cudaMalloc(&d_low_bound, sizeof(fr_t));
    cudaMemcpy(d_low_bound, &low_bound, sizeof(fr_t), cudaMemcpyHostToDevice);
    tlookuprange_init_kernel<<<(len + 256 - 1) / 256, 256>>>(T.gpu_data, B.gpu_data, d_low_bound, len, d_beta);
    cudaDeviceSynchronize();
}

void tLookupRange::prep(const FrTensor& vals){
    uint* hist;
    cudaMalloc((void **)&hist, sizeof(uint) * m.size);
    cudaMemset(hist, 0, sizeof(uint) * m.size);
    Fr h_low = Fr(low);
    fr_t *d_low;
    cudaMalloc(&d_low, sizeof(fr_t));
    cudaMemcpy(d_low, &h_low, sizeof(fr_t), cudaMemcpyHostToDevice);

    get_histogram_kernel<<<(vals.size + 255) / 256, 256>>>(vals.gpu_data, hist, vals.size, d_low);
    cudaDeviceSynchronize();
    if(vals.size != 1 << Log2(vals.size)){
        uint zero_count;
        cudaMemcpy(&zero_count, hist + (-low), sizeof(uint), cudaMemcpyDeviceToHost);
        zero_count += (1 << Log2(vals.size)) - vals.size;
        cudaMemcpy(hist + (-low), &zero_count, sizeof(uint), cudaMemcpyHostToDevice);
        //cout << "zero_count: " << zero_count << endl;
    }
    m.get_data(m.size, hist);
    cudaFree(hist);
    cudaFree(d_low);
}

Fr tLookupRange::prove(FrTensor& S, FrTensor& A, const vector<Fr>& v, 
                            Hyrax &hyrax, uint msm_size, double& commit_time)
{ 
    uint D = S.size;
    const uint N = T.size;
    auto u = random_vec(Log2(D));

    assert(u.size() == v.size());

    prep(S);

    if (D != 1 << Log2(D))
    {
        Fr pad_value = Fr::zero();
        S.pad(A, 11008, 16384, 1 << Log2(D), pad_value);
        D = 1 << Log2(D);
        cudaDeviceSynchronize();
    }
    A.set_size(D);

    assert(D % msm_size == 0);

    if (v.size() != Log2(D)) throw std::runtime_error("u.size() != ceilLog2(D)");

    fr_t *d_beta;
    cudaMalloc(&d_beta, sizeof(fr_t));
    cudaMemcpy(d_beta, &beta, sizeof(fr_t), cudaMemcpyHostToDevice);
    tlookup_inv_kernel<<<(D + 255) / 256, 256>>>(S.gpu_data, A.gpu_data, d_beta, D);
    cudaDeviceSynchronize();
    A.set_size(D);
    
    timer.start();
    hyrax.commit(A, D, msm_size);
    hyrax.commit(m, N, 1 << (Log2(N) / 2));
    CUDA_DEBUG;
    commit_time += timer.stop("commit A m");

    FrTensor Bm(B);   Bm *= m; 
    Fr C = alpha * alpha - Bm.sum(N);

    Fr alpha_sq = alpha * alpha;
    Fr claim = alpha + alpha_sq;

    Fr N_div_D = Fr(N) * (Fr(D).invert());
    
    vector<Fr> v1 = {v.begin() + Log2(N), v.end()};
    vector<Fr> v2 = {v.begin(), v.begin() + Log2(N)};

    
    Fr S_claim = tLookup_phase1(claim, A, S, B, T, m, 
        alpha, beta, C, N_div_D, alpha_sq, 
        u, v1, v2);
    
    return S_claim;
    
}

tLookupRange::~tLookupRange()
{
}


tLookupRangeMapping::tLookupRangeMapping(const int low, const uint len, const string& filename) : low(low),
    tLookup(len), mapped_vals(len), r(Fr::random_element())
{
    assert(len == 1 << Log2(len));

    int* int_gpu_data;
    cudaMalloc((void **)&int_gpu_data, sizeof(int) * len);
    loadbin(filename, int_gpu_data, sizeof(int) * len);
    int_to_fr<<<(len+256-1)/256,256>>>(int_gpu_data, mapped_vals.gpu_data, len);
    cudaDeviceSynchronize();

    fr_t *d_r; cudaMalloc(&d_r, sizeof(fr_t));  cudaMemcpy(d_r, &r, sizeof(fr_t), cudaMemcpyHostToDevice);
    Fr h_low = Fr(low);
    fr_t *d_low; cudaMalloc(&d_low, sizeof(fr_t));  cudaMemcpy(d_low, &h_low, sizeof(fr_t), cudaMemcpyHostToDevice);
    fr_t *d_beta; cudaMalloc(&d_beta, sizeof(fr_t));  cudaMemcpy(d_beta, &beta, sizeof(fr_t), cudaMemcpyHostToDevice);
    tLookupRangeMapping_init_kernel<<<(len + 256 - 1) / 256, 256>>>(T.gpu_data, B.gpu_data, mapped_vals.gpu_data, d_low, len, d_r, d_beta);
    cudaDeviceSynchronize();
    cudaFree(int_gpu_data);
    cudaFree(d_r);
    cudaFree(d_low);
    cudaFree(d_beta);
}

void tLookupRangeMapping::prep(const FrTensor& vals){
    uint* hist;
    cudaMalloc((void **)&hist, sizeof(uint) * m.size);
    cudaMemset(hist, 0, sizeof(uint) * m.size);
    Fr h_low = Fr(low);
    fr_t *d_low;
    cudaMalloc(&d_low, sizeof(fr_t));
    cudaMemcpy(d_low, &h_low, sizeof(fr_t), cudaMemcpyHostToDevice);

    get_histogram_kernel<<<(vals.size + 255) / 256, 256>>>(vals.gpu_data, hist, vals.size, d_low);
    cudaDeviceSynchronize();

    if(vals.size != 1 << Log2(vals.size)){
        uint zero_count;
        cudaMemcpy(&zero_count, hist + (-low), sizeof(uint), cudaMemcpyDeviceToHost);
        zero_count += (1 << Log2(vals.size)) - vals.size;
        cudaMemcpy(hist + (-low), &zero_count, sizeof(uint), cudaMemcpyHostToDevice);
        //cout << "zero_count: " << zero_count << endl;
    }
    m.get_data(m.size, hist);
    cudaFree(hist);
    cudaFree(d_low);
}

Fr tLookupRangeMapping::prove(FrTensor& S_in, FrTensor& S_out, FrTensor& A, const vector<Fr>& v, 
                                Hyrax &hyrax, uint msm_size, double& commit_time)
{
    uint D = S_in.size;
    uint N = m.size;

    auto u = random_vec(Log2(D));
    assert(u.size() == v.size());

    prep(S_in);

    FrTensor S(1 << Log2(D));  //因为接下来可能要pad，所以大小设置成这个
    S = S_out;
    S *= r;
    S.add_with_size(D, S_in);
   
    if (D != 1 << Log2(D))
    {
        Fr pad_value = mapped_vals(uint(-low)) * r;
        S.pad(A, 11008, 16384, 1 << Log2(D), pad_value);
        D = 1 << Log2(D);
    }
    A.set_size(D);

    fr_t *d_beta;  cudaMalloc(&d_beta, sizeof(fr_t));  cudaMemcpy(d_beta, &beta, sizeof(fr_t), cudaMemcpyHostToDevice);
    
    tlookup_inv_kernel<<<(D + 255) / 256, 256>>>(S.gpu_data, A.gpu_data, d_beta, D);
    cudaDeviceSynchronize();


    assert(D % msm_size == 0);

    timer.start();
    hyrax.commit(A, D, msm_size);    
    hyrax.commit(m, N, 1 << (Log2(N) / 2));
    CUDA_DEBUG;
    commit_time += timer.stop("commit A m");
    
    
    if (v.size() != Log2(D)) throw std::runtime_error("v.size() != ceilLog2(D)");

    FrTensor Bm(B);   Bm *= m; 
    Fr C = alpha * alpha - Bm.sum(N);

    Fr alpha_sq = alpha * alpha;
    Fr claim = alpha + alpha_sq;

    Fr N_div_D = Fr(N) * (Fr(D).invert());
    
    vector<Fr> v1 = {v.begin() + Log2(N), v.end()};
    vector<Fr> v2 = {v.begin(), v.begin() + Log2(N)};

    
    Fr S_claim = tLookup_phase1(claim, A, S, B, T, m, 
        alpha, beta, C, N_div_D, alpha_sq, 
        u, v1, v2);
        
    return S_claim;

}