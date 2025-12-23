#pragma once
#include "./field/alt_bn128.cuh"
#include "./fr-tensor.cuh"
#include "./ioutils.cuh"
#include "ec_operation.cuh"
#include "polynomial.cuh"
using namespace alt_bn128;

struct MatMulSumcheckWorkTmp {
    FrTensor tmpw;
    FrTensor a;
    FrTensor b;
    FrTensor c;

    size_t layer_num;

    MatMulSumcheckWorkTmp(size_t layer_num)
        : layer_num(layer_num),
          tmpw(11008),
          a(layer_num * 11008 / 2),
          b(layer_num * 11008 / 2),
          c(layer_num * 11008 / 2)
    {}
    // 禁止拷贝，避免误拷贝 GPU 内存
    MatMulSumcheckWorkTmp(const MatMulSumcheckWorkTmp&) = delete;
    MatMulSumcheckWorkTmp& operator=(const MatMulSumcheckWorkTmp&) = delete;
};


__global__ void sumcheck_kernel(fr_t *A, fr_t *B, fr_t *a, fr_t *b, fr_t *c, int in_size, int out_size, fr_t *v){
    uint gid = GET_GLOBAL_ID();
    if(gid > out_size) return;

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
Fr matmul_sumcheck_phase1(FrTensor &A, FrTensor &B, FrTensor &a, FrTensor &b, FrTensor &c, 
    fr_t *u, fr_t *v, uint K, uint size, Fr &claim){

    uint rounds = Log2(size / K);
    uint inputsize = size;
    uint outputsize = inputsize / 2;
    Fr u_now; 
    Fr v_now;
    Fr eq_accumulate = Fr::one();
    for(int i = 0; i < rounds; i++){
        sumcheck_kernel<<<(outputsize + 255) / 256, 256>>>(A.gpu_data, B.gpu_data, a.gpu_data, b.gpu_data, c.gpu_data, inputsize, outputsize, v + Log2(size) - i - 1);
        
        a.partial_eval(outputsize, u, outputsize / K, 0, K);
        b.partial_eval(outputsize, u, outputsize / K, 0, K);
        c.partial_eval(outputsize, u, outputsize / K, 0, K);

        
        Polynomial p({c.sum(K), b.sum(K), a.sum(K)});
        cudaMemcpy(&u_now, u + rounds - i - 1, sizeof(fr_t), cudaMemcpyDeviceToHost);
        //CUDA_DEBUG;
        p *= Polynomial::eq(eq_accumulate, u_now);

        if(p(0) + p(1) != claim) throw std::runtime_error("matmul_phase1: claim != p(0) + p(1)");
        
        cudaMemcpy(&v_now, v + Log2(size) - i - 1, sizeof(fr_t), cudaMemcpyDeviceToHost);
        claim = p(v_now);
    
        // A.partial_eval(inputsize, v, 2, Log2(size) - i - 1, outputsize);
        // B.partial_eval(inputsize, v, 2, Log2(size) - i - 1, outputsize);
  
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
    printf("%u %u %u\n", inputsize, outputsize, rounds);
    Fr v_now;
    for(int i = 0; i < rounds; i++){
        sumcheck_kernel<<<(outputsize + 255) / 256, 256>>>(A.gpu_data, B.gpu_data, a.gpu_data, b.gpu_data, c.gpu_data, inputsize, outputsize, v + Log2(K) - i - 1);
        Polynomial p({c.sum(outputsize) * eq_accumulate, b.sum(outputsize) * eq_accumulate, a.sum(outputsize) * eq_accumulate});

        if(p(0) + p(1)!= claim) throw std::runtime_error("matmul_phase1: claim != p(0) + p(1)");

        cudaMemcpy(&v_now, v + Log2(K) - i - 1, sizeof(fr_t), cudaMemcpyDeviceToHost);
        claim = p(v_now);
    
        // A.partial_eval(inputsize, v, 2, Log2(K) - i - 1, outputsize);
        // B.partial_eval(inputsize, v, 2, Log2(K) - i - 1, outputsize);
        inputsize = outputsize;
        outputsize = outputsize / 2;

    }
}

void prove_matmul(FrTensor &A, FrTensor &B, MatMulSumcheckWorkTmp &tmp, int L, int N, int M, int K, const vector<Fr> u, const vector<Fr> v, Fr &claim){
    fr_t *U;   cudaMalloc(&U, sizeof(fr_t) * u.size());    cudaMemcpy(U, u.data(), sizeof(fr_t) * u.size(), cudaMemcpyHostToDevice);
    fr_t *V;   cudaMalloc(&V, sizeof(fr_t) * v.size());    cudaMemcpy(V, v.data(), sizeof(fr_t) * v.size(), cudaMemcpyHostToDevice);

    A.partial_eval(A.size, U, N, Log2(M), K);
    B.partial_eval(B.size, U, M, 0, K);
    
 
    CPU_TIMER_START(matmul_phase1);
    Fr eq_accumulate = matmul_sumcheck_phase1(A, B, tmp.a, tmp.b, tmp.c, U + Log2(N * M), V, K, K * L, claim);
    CUDA_DEBUG;
    CPU_TIMER_STOP(matmul_phase1);

    tmp.tmpw = B;
    CPU_TIMER_START(matmul_phase2);
    matmul_sumcheck_phase2(A, B, tmp.a, tmp.b, tmp.c, V, K, claim, eq_accumulate);
    CUDA_DEBUG;
    CPU_TIMER_STOP(matmul_phase2);

    Fr A_claim , B_claim;
    cudaMemcpy(&A_claim, A.gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&B_claim, B.gpu_data, sizeof(fr_t), cudaMemcpyDeviceToHost);
    
    assert(A_claim * B_claim * eq_accumulate == claim);
    cudaFree(V);
    cudaFree(U);
}

