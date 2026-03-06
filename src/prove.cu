#include "prove.cuh"

// 结构体方法实现
SumcheckWorkTmp::SumcheckWorkTmp(size_t tokens_size_)
    : tokens_size(tokens_size_),
      tmpw(11008),
      a(tokens_size_ * (1 << 13)),
      b(tokens_size_ * (1 << 13)),
      c(tokens_size_ * (1 << 13))
{
}

// 内核函数实现-----------------------------------------------------------------------------------------
__global__ void sumcheck_kernel(fr_t *X, fr_t *a, fr_t *b, 
                                int in_size, int out_size, fr_t *v)
{
    uint gid = GET_GLOBAL_ID();
    if(gid >= out_size) return;

    fr_t Xa = (gid + out_size) < in_size ? X[gid + out_size] - X[gid] : fr_t::cneg(X[gid], true);
    fr_t Xb = X[gid];

    X[gid] = Xa * v[0] + Xb;
    
    b[gid] = Xb;
    a[gid] = Xa;
}

__global__ void sumcheck_kernel(fr_t *A, fr_t *B, fr_t *a, fr_t *b, fr_t *c, 
                                int in_size, int out_size, fr_t *v)
{
    uint gid = GET_GLOBAL_ID();
    if(gid >= out_size) return;

    fr_t Aa = (gid + out_size) < in_size ? A[gid + out_size] - A[gid] : fr_t::cneg(A[gid], true);
    fr_t Ab = A[gid];
    fr_t Ba = (gid + out_size) < in_size ? B[gid + out_size] - B[gid] : fr_t::cneg(B[gid], true);
    fr_t Bb = B[gid];

    A[gid] = Aa * v[0] + Ab;
    B[gid] = Ba * v[0] + Bb;

    c[gid] = Ab * Bb;
    b[gid] = Aa * Bb + Ab * Ba;
    a[gid] = Aa * Ba;
}

// 普通函数实现----------------------------------------------------------------------------------------------------
Fr matmul_sumcheck_phase1(FrTensor &A, FrTensor &B, FrTensor &a, FrTensor &b, FrTensor &c, 
                         fr_t *u, fr_t *v, uint K, uint size, Fr &claim)
{
    uint rounds = Log2(size / K);
    uint inputsize = size;
    uint outputsize = inputsize / 2;
    Fr u_now; 
    Fr v_now;
    Fr eq_accumulate = Fr::one();
    
    for(int i = 0; i < rounds; i++){
        sumcheck_kernel<<<(outputsize + 255) / 256, 256>>>(
            A.gpu_data, B.gpu_data, a.gpu_data, b.gpu_data, c.gpu_data, 
            inputsize, outputsize, v - i - 1);
        
        a.partial_eval(outputsize, u - rounds, outputsize / K, 0, K);
        b.partial_eval(outputsize, u - rounds, outputsize / K, 0, K);
        c.partial_eval(outputsize, u - rounds, outputsize / K, 0, K);

        Polynomial p({c.sum(K), b.sum(K), a.sum(K)});
        cudaMemcpy(&u_now, u - i - 1, sizeof(fr_t), cudaMemcpyDeviceToHost);
        p *= (Polynomial::eq(u_now) * eq_accumulate);

        if(p(0) + p(1) != claim) {
            throw std::runtime_error("matmul_phase1: claim != p(0) + p(1)");
        }
        
        cudaMemcpy(&v_now, v - i - 1, sizeof(fr_t), cudaMemcpyDeviceToHost);
        claim = p(v_now);
    
        inputsize = outputsize;
        outputsize = outputsize / 2;
        eq_accumulate *= (Fr::one() - v_now) * (Fr::one() - u_now) + v_now * u_now;
    }
    return eq_accumulate;
}

void matmul_sumcheck_phase2(FrTensor &A, FrTensor &B, FrTensor &a, FrTensor &b, FrTensor &c, 
                           fr_t *v, uint K, Fr &claim, Fr &eq_accumulate)
{
    uint rounds = Log2(K);
    uint inputsize = K;
    uint outputsize = 1 << (rounds - 1);
    Fr v_now;
    
    for(int i = 0; i < rounds; i++){
        sumcheck_kernel<<<(outputsize + 255) / 256, 256>>>(
            A.gpu_data, B.gpu_data, a.gpu_data, b.gpu_data, c.gpu_data, 
            inputsize, outputsize, v - i - 1);
        
        Polynomial p({
            c.sum(outputsize) * eq_accumulate, 
            b.sum(outputsize) * eq_accumulate, 
            a.sum(outputsize) * eq_accumulate
        });

        if(p(0) + p(1) != claim) {
            throw std::runtime_error("matmul_phase2: claim != p(0) + p(1)");
        }

        cudaMemcpy(&v_now, v - i - 1, sizeof(fr_t), cudaMemcpyDeviceToHost);
        claim = p(v_now);
    
        inputsize = outputsize;
        outputsize = outputsize / 2;
    }
}

Fr eleMul_sumcheck(FrTensor &A, FrTensor &B, FrTensor &a, FrTensor &b, FrTensor &c, 
                  fr_t *u, fr_t *v, uint first_size, uint size, Fr &claim)
{
    if((1 << Log2(first_size)) != first_size) {
        throw std::runtime_error("eleMul_sumcheck: first_size is illegal");
    }
    
    uint second_size = size / first_size;
    uint rounds = Log2(size);
    uint inputsize = size;
    uint outputsize = size / 2;
    Fr u_now; 
    Fr v_now;
    Fr eq_accumulate = Fr::one();
    
    for(int i = 0; i < rounds; i++){
        sumcheck_kernel<<<(outputsize + 255) / 256, 256>>>(
            A.gpu_data, B.gpu_data, a.gpu_data, b.gpu_data, c.gpu_data, 
            inputsize, outputsize, v - i - 1);
            
        if(inputsize > second_size){
            a.partial_eval(outputsize, u - Log2(first_size), outputsize / second_size, 0, second_size);
            a.partial_eval(second_size, u - Log2(size), second_size, 0, 1);

            b.partial_eval(outputsize, u - Log2(first_size), outputsize / second_size, 0, second_size);
            b.partial_eval(second_size, u - Log2(size), second_size, 0, 1);

            c.partial_eval(outputsize, u - Log2(first_size), outputsize / second_size, 0, second_size);
            c.partial_eval(second_size, u - Log2(size), second_size, 0, 1);
        }
        else{
            a.partial_eval(outputsize, u - Log2(size), outputsize, 0, 1);
            b.partial_eval(outputsize, u - Log2(size), outputsize, 0, 1);
            c.partial_eval(outputsize, u - Log2(size), outputsize, 0, 1);
        }
        
        Polynomial p({c(0), b(0), a(0)});
        cudaMemcpy(&u_now, u - i - 1, sizeof(fr_t), cudaMemcpyDeviceToHost);
        p *= (Polynomial::eq(u_now) * eq_accumulate);

        if(p(0) + p(1) != claim) {
            throw std::runtime_error("ele_Mul: claim != p(0) + p(1)");
        }
        
        cudaMemcpy(&v_now, v - i - 1, sizeof(fr_t), cudaMemcpyDeviceToHost);
        claim = p(v_now);
    
        inputsize = outputsize;
        outputsize = outputsize == second_size ? (1 << (Log2(outputsize) - 1)) : outputsize / 2;
        eq_accumulate *= ((Fr::one() - v_now) * (Fr::one() - u_now) + v_now * u_now);
    }
    return eq_accumulate;
}

// 证明函数实现
std::pair<Fr, Fr> prove_matmul(FrTensor &A, FrTensor &B, 
                  int L, int N, int M, int K, 
                  const std::vector<Fr> u, const std::vector<Fr> v, Fr &claim)
{
    fr_t *U;   
    cudaMalloc(&U, sizeof(fr_t) * u.size());    
    cudaMemcpy(U, u.data(), sizeof(fr_t) * u.size(), cudaMemcpyHostToDevice);
    
    fr_t *V;   
    cudaMalloc(&V, sizeof(fr_t) * v.size());    
    cudaMemcpy(V, v.data(), sizeof(fr_t) * v.size(), cudaMemcpyHostToDevice);

    A.partial_eval(A.size, U, N, Log2(M), K);
    B.partial_eval(B.size, U, M, 0, K);
    
    FrTensor a(L * K);
    FrTensor b(L * K);
    FrTensor c(L * K);
    Fr eq_accumulate = matmul_sumcheck_phase1(A, B, a, b, c, 
                                             U + Log2(L * N * M), V + Log2(L * K), 
                                             K, K * L, claim);
    //tmp.tmpw = B;

    matmul_sumcheck_phase2(A, B, a, b, c, 
                          V + Log2(K), K, claim, eq_accumulate);

    Fr A_claim, B_claim;
    cudaMemcpy(&A_claim, A.gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&B_claim, B.gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);
    
    assert(A_claim * B_claim * eq_accumulate == claim);
    
    cudaFree(V);
    cudaFree(U);
    return {A_claim, B_claim};
}

std::pair<Fr, Fr> prove_eleMul(FrTensor &A, FrTensor &B, 
                  int L, int M, int N, 
                  const std::vector<Fr> u, const std::vector<Fr> v, Fr &claim)
{
    fr_t *U;   
    cudaMalloc(&U, sizeof(fr_t) * u.size());    
    cudaMemcpy(U, u.data(), sizeof(fr_t) * u.size(), cudaMemcpyHostToDevice);
    
    fr_t *V;   
    cudaMalloc(&V, sizeof(fr_t) * v.size());    
    cudaMemcpy(V, v.data(), sizeof(fr_t) * v.size(), cudaMemcpyHostToDevice);

    FrTensor a(L * M * N / 2);
    FrTensor b(L * M * N / 2);
    FrTensor c(L * M * N / 2);
    
    Fr eq_accumulate = eleMul_sumcheck(A, B, a, b, c, 
                                      U + Log2(L * N * M), V + Log2(L * M * N), 
                                      L * M, L * M * N, claim);
                                      
    Fr A_claim, B_claim;
    cudaMemcpy(&A_claim, A.gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&B_claim, B.gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);
    
    assert(A_claim * B_claim * eq_accumulate == claim);
    
    cudaFree(V);
    cudaFree(U);
    return {A_claim, B_claim};
}

Fr combine_claims(FrTensor &X, const vector<Fr> &claims, 
                const vector<vector<Fr>> &u, const vector<Fr> &v, const uint size){

    assert(size == (1 << Log2(size)));
    uint num = claims.size();
    for(int i = 0; i < num; i++) assert(u[i].size() == v.size());
    assert((1 << v.size()) == size);
    Fr claim = Fr::zero();
    for(int i = 0; i < num; i++) claim += claims[i];
    uint rounds = v.size();
    FrTensor a(size / 2), b(size / 2);
    FrTensor tmp_a(size / 2), tmp_b(size / 2);
    uint input_size = size;
    uint output_size = size / 2;

    vector<vector<Fr>> new_u = u;
    vector<Fr> new_v = v;
    fr_t *d_v;  cudaMalloc(&d_v, sizeof(fr_t));
    vector<Fr> eq_accumulate(num, Fr::one());
    for(uint i = 0; i < rounds; i++){ 
        a.set_size(output_size);
        b.set_size(output_size);
        tmp_a.set_size(output_size);
        tmp_b.set_size(output_size);
        cudaMemcpy(d_v, &new_v.back(), sizeof(fr_t), cudaMemcpyHostToDevice);

        sumcheck_kernel<<<(output_size + 255) / 256, 256>>>(X.gpu_data, a.gpu_data, b.gpu_data, input_size, output_size, d_v);

        Polynomial p(2);
        for(uint j = 0; j < num; j++){
            Fr now_u = new_u[j].back();
            new_u[j].pop_back();
            tmp_a = a;
            tmp_b = b;
            //assert(tmp_a.size == 1 << new_u[j].size());
            Polynomial p1({tmp_b(new_u[j]), tmp_a(new_u[j])}); 
            p1 *= Polynomial::eq(now_u);
            p1 *= eq_accumulate[j];
            p += p1;
            eq_accumulate[j] *= Polynomial::eq(now_u, new_v.back());
        }
        assert(p(0) + p(1) == claim);
        claim = p(new_v.back());

        new_v.pop_back();

        input_size = output_size;
        output_size = output_size / 2;

    }

    Fr ret = X(0);
    Fr sum = Fr::zero();
    for(uint i = 0; i < num; i++) sum += eq_accumulate[i] * ret;
    assert(sum == claim);
    cudaFree(d_v);
    return ret;
}