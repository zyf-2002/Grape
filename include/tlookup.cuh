#pragma once
#include "ec_operation.cuh"
#include "fr-tensor.cuh" 
#include "polynomial.cuh"
#include "ioutils.cuh"
#include "Hyrax.cuh"
#include "timer.cuh"

// 内核函数声明
__global__ void get_histogram_kernel(const fr_t* value, uint *hist, uint size, fr_t *low);
__global__ void tlookuprange_init_kernel(fr_t* table_ptr, fr_t* inv_table_ptr, const fr_t *low, 
                                                const uint table_size, const fr_t *beta);
__global__ void tlookup_inv_kernel(fr_t* value, fr_t* value_inv, fr_t *beta, uint size);
__global__ void tLookup_poly_kernel(const fr_t* A_data, const fr_t* S_data, fr_t *beta, fr_t* out0, fr_t* out1, fr_t* out2, uint N_out);
__global__ void tLookup_poly_sum_kernel(const fr_t* A_data, fr_t* out0, fr_t* out1, uint N_out);
__global__ void tLookup_phase1_reduce_kernel(const fr_t* A_data, const fr_t* S_data, fr_t* new_A_data, fr_t* new_S_data, fr_t *v, uint N_out);
__global__ void tLookup_phase2_reduce_kernel(const fr_t* A_data, const fr_t* S_data, const fr_t* B_data, const fr_t* T_data, const fr_t* m_data,
    fr_t* new_A_data, fr_t* new_S_data, fr_t* new_B_data, fr_t* new_T_data, fr_t* new_m_data,
    fr_t *v, uint N_out);

// 辅助函数声明
Polynomial tLookup_phase1_step_poly(const Fr& claim, const FrTensor& A, const FrTensor& S, 
    const Fr& alpha, const Fr& beta, const Fr& C, const vector<Fr>& u);
    
Polynomial tLookup_phase2_step_poly(const FrTensor& A, const FrTensor& S, const FrTensor& B, const FrTensor& T, const FrTensor& m,
    const Fr& alpha_, const Fr& beta, const Fr& inv_size_ratio, const Fr& alpha_sq,
    const vector<Fr>& u);
    
Fr tLookup_phase2(const Fr& claim, const FrTensor& A, const FrTensor& S, const FrTensor& B, const FrTensor& T, const FrTensor& m,
    const Fr& alpha_, const Fr& beta, const Fr& inv_size_ratio, const Fr& alpha_sq, const vector<Fr>& u, const vector<Fr>& v2);
    
Fr tLookup_phase1(const Fr& claim, FrTensor& A, FrTensor& S, const FrTensor& B, const FrTensor& T, FrTensor& m,
    const Fr& alpha, const Fr& beta, const Fr& C, const Fr& inv_size_ratio, const Fr& alpha_sq, 
    const vector<Fr>& u, const vector<Fr>& v1, const vector<Fr>& v2);

// 类声明
class tLookup
{
    public:
    Fr alpha, beta;
    FrTensor T;
    FrTensor B;
    FrTensor m;
    
    tLookup(const uint size);
    
    // We do not directly use the values from the tensors. Instead, we assume that the tensors have been elementwisely converted to the indices of the table.
    FrTensor prep(const uint* indices, const uint D); // D - dimension of the tensor

    
};

class tLookupRange: public tLookup
{
    public:
    const int low;
    
    tLookupRange(int low, uint len);
    ~tLookupRange();
    
    void prep(const FrTensor& vals);

    Fr prove(FrTensor& S,  FrTensor& A, const vector<Fr>& v, 
                Hyrax &hyrax, uint msm_size, double& commit_time);
};

class tLookupRangeMapping : public tLookup
{
    public:
    Fr r;
    int low;
    FrTensor mapped_vals;
    tLookupRangeMapping(const int low, const uint len, const string& filename);

    void prep(const FrTensor& vals);
    
    Fr prove(FrTensor& S_in, FrTensor& S_out, FrTensor &A, const vector<Fr>& v, 
                Hyrax &hyrax, uint msm_size, double& commit_time);
};