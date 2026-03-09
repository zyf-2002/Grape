#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include "fr-tensor.cuh"
#include "Hyrax.cuh"
#include "ec_operation.cuh"
#include "prove.cuh"
#include "tlookup.cuh"
#include "timer.cuh"
#include <omp.h>

using namespace std;
using namespace libsnark;


uint max_embed_dim = 11008;
uint dim = 4096;
uint npoints = 16384;

int main(int argc, char* argv[])
{
    if (argc < 3) {
        return 1;
    }
    uint layer_num = atoi(argv[1]);
    uint input_tokens = atoi(argv[2]);
    cout << "layer_num: " << layer_num << endl;
    cout << "input_tokens: " << input_tokens << endl;

    ppT::init_public_params();
    affine_t *points;
    cudaMalloc((void **)&points, npoints * layer_num * sizeof(affine_t) * 256);
    loadbin("../data/points.bin", points, npoints * layer_num * sizeof(affine_t) * 256);
    vector<bn128> cpu_points(npoints * layer_num);
    loadbin("../data/cpu_points.bin", cpu_points.data(), npoints * layer_num * sizeof(bn128), false);

    int *read_tensor1 = nullptr;
    int *read_tensor2 = nullptr;
    uint data_size1;
    uint data_size2;
    FrTensor tensor1(layer_num * dim * max_embed_dim);
    FrTensor tensor2(layer_num * dim * max_embed_dim);
    FrTensor tensor3(layer_num * input_tokens * (1 << Log2(max_embed_dim)));
    FrTensor tmp_tensor(layer_num * input_tokens * dim);

    vector<Fr> eval_point1;
    vector<Fr> eval_point2;
    vector<Fr> eval_point3;
    vector<Fr> skip_eval_point;
    Fr c = Fr::random_element();
    Hyrax hyrax(layer_num, npoints, points, cpu_points[0]);

    uint SCALE1 = 1 << 16;
    uint SCALE2 = 1 << 20;
    tLookupRange rs(-(SCALE1 / 2), SCALE1);
    tLookupRange swiglu_rs(-(1 << 19), 1 << 20);
    tLookupRange softmax_range_relation(0, 1 << 16);
    tLookupRange exp_rs(-(1 << 21), 1 << 22);


    tLookupRangeMapping swiglu(-(1 << 18), 1 << 19, "../data/table/swiglu-table.bin");
    tLookupRangeMapping exp(-(1 << 20) + 1, 1 << 20, "../data/table/exp-table.bin");

    CUDA_DEBUG;
    //-----------------------------------------------------------prepare done---------------------------------------------------
    
    Timer timer;
    double total_prove_time = 0;

for(int layer_id = (32 / layer_num) - 1; layer_id >= 0; layer_id--)
{
    data_size1 = load_data("../data/Q/output-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);

    generate_random_eval_points(data_size1, eval_point1);
    auto proof = hyrax.open(tensor1, eval_point1, c, data_size1, data_size1 / layer_num / input_tokens);
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");

    
    data_size1 = load_data("../data/Q/down_out-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor1(eval_point1);

    skip_eval_point = eval_point1;

    Fr down_out_value = tensor1(0);
    Fr skip_value = proof.result - down_out_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");
    // ------------------------------------lock skip_eval_point-----------------------------------

    data_size1 = load_data("../data/R/down_out-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor3.set_size(data_size1);   tensor3 = tensor1;

    Fr lookup_value = rs.prove(tensor1, tensor2, eval_point1, hyrax, dim);
    proof = hyrax.open(tensor3, eval_point1, c, data_size1, data_size1 / layer_num / input_tokens);
    assert(proof.result == lookup_value);
    down_out_value = down_out_value * SCALE1 + lookup_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");


    data_size1 = load_data("../data/Q/upSilu-" + to_string(layer_id) + ".bin", &read_tensor1);
    data_size2 = load_data("../data/W/down-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    generate_random_eval_points(max_embed_dim * layer_num, eval_point2);
    tensor3.get_data(data_size1, read_tensor1);
    tensor2.get_data(data_size2, read_tensor2);
    tensor1.set_size(data_size2);   tensor1 = tensor2;
    
    auto down_pair = prove_matmul(tensor3, tensor2, layer_num, input_tokens, dim, max_embed_dim, eval_point1, eval_point2, down_out_value);
    ReassembleVectors(eval_point1, eval_point2, Log2(input_tokens), Log2(max_embed_dim), Log2(dim));
    proof = hyrax.open(tensor1, eval_point2, c, data_size2, max_embed_dim);
    assert(proof.result == down_pair.second);
    Fr upSilu_value = down_pair.first;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");


    data_size1 = load_data("../data/R/upSilu-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor3.set_size(data_size1);   tensor3 = tensor1;
    lookup_value = rs.prove(tensor1, tensor2, eval_point1, hyrax, npoints);
    proof = hyrax.open(tensor3, eval_point1, c, data_size1, data_size1 / layer_num / input_tokens);
    assert(proof.result == lookup_value);
    upSilu_value = upSilu_value  * SCALE1 + lookup_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");

    data_size1 = load_data("../data/Q/up_out-" + to_string(layer_id) + ".bin", &read_tensor1);
    data_size2 = load_data("../data/Q/silu_out-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor2.get_data(data_size2, read_tensor2);
    generate_random_eval_points(data_size2, eval_point2);
    tensor3.set_size(data_size2);   tensor3 = tensor2;
    
    auto upSilu_pair = prove_eleMul(tensor1, tensor2, layer_num, input_tokens, max_embed_dim, eval_point1, eval_point2, upSilu_value);
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");
    // ------------------------------------lock eval_point2-----------------------------------

    data_size1 = load_data("../data/Q/gate_out-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor2.set_size(data_size1);
    generate_random_eval_points(data_size1, eval_point1);
    lookup_value = swiglu.prove(tensor1, tensor3, tensor2, eval_point1, hyrax, npoints);

    proof = hyrax.open(tensor1, eval_point1, c, data_size1, max_embed_dim);
    Fr gate_out_value = proof.result;
    proof = hyrax.open(tensor3, eval_point1, c, data_size1, max_embed_dim);
    assert(lookup_value == gate_out_value + proof.result * swiglu.r);
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");
    // ------------------------------------lock eval_point1-----------------------------------
    // -------------------------------unlock eval_point2----------------------------------
    data_size1 = load_data("../data/R/up_out-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor3.set_size(data_size1);   tensor3 = tensor1;
    lookup_value = rs.prove(tensor1, tensor2, eval_point2, hyrax, npoints);
    proof = hyrax.open(tensor3, eval_point2, c, data_size1, data_size1 / layer_num / input_tokens);
    assert(proof.result == lookup_value);
    Fr up_out_value = upSilu_pair.first * SCALE1 + lookup_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop("up_out_lookup");


    data_size1 = load_data("../data/Q/normOutSecond-" + to_string(layer_id) + ".bin", &read_tensor1);
    data_size2 = load_data("../data/W/up-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    generate_random_eval_points(dim * layer_num, eval_point3);
    tensor3.get_data(data_size1, read_tensor1);
    tensor2.get_data(data_size2, read_tensor2);
    tensor1.set_size(data_size2);  tensor1 = tensor2;
    tmp_tensor = tensor3;
    
    auto up_pair = prove_matmul(tensor3, tensor2, layer_num, input_tokens, max_embed_dim, dim, eval_point2, eval_point3, up_out_value);
    ReassembleVectors(eval_point2, eval_point3, Log2(input_tokens), Log2(dim), Log2(max_embed_dim));
    proof = hyrax.open(tensor1, eval_point3, c, data_size2, dim);
    
    assert(proof.result == up_pair.second);
    Fr norm_second_out_value1 = up_pair.first;
   
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");
    // -------------------------------lock eval_point2----------------------------------
    // -------------------------------lock tmp_tensor----------------------------------
    // -------------------------------unlock eval_point1----------------------------------
    
    data_size1 = load_data("../data/R/gate_out-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor3.set_size(data_size1);  tensor3 = tensor1;
    lookup_value = swiglu_rs.prove(tensor1, tensor2, eval_point1, hyrax, npoints);
    proof = hyrax.open(tensor3, eval_point1, c, data_size1, data_size1 / layer_num / input_tokens);
    assert(proof.result == lookup_value);
    gate_out_value = gate_out_value * SCALE2 + lookup_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");

    // -------------------------------unlock tmp_tensor----------------------------------
    data_size2 = load_data("../data/W/gate-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    tensor3 = tmp_tensor;
    generate_random_eval_points(dim * layer_num, eval_point3);
    tensor2.get_data(data_size2, read_tensor2);
    tensor1.set_size(data_size2);   tensor1 = tensor2;

    auto gate_pair = prove_matmul(tensor3, tensor2, layer_num, input_tokens, max_embed_dim, dim, eval_point1, eval_point3, gate_out_value);

    ReassembleVectors(eval_point1, eval_point3, Log2(input_tokens), Log2(dim), Log2(max_embed_dim));
    proof = hyrax.open(tensor1, eval_point3, c, data_size2, dim);
    assert(proof.result == gate_pair.second);
    Fr norm_second_out_value2 = gate_pair.first;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");
    // -------------------------------unlock eval_point2----------------------------------
   
    timer.start();
    generate_random_eval_points(dim * layer_num * input_tokens, eval_point3);
    Fr normSecondOut_value = combine_claims(tmp_tensor, {norm_second_out_value1, norm_second_out_value2}, 
                                {eval_point2, eval_point1}, eval_point3, dim * layer_num * input_tokens);
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");

// ================================norm begin===========================================
    data_size1 = load_data("../data/R/normOutSecond-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor3.set_size(data_size1);  tensor3 = tensor1;
    lookup_value = rs.prove(tensor1, tensor2, eval_point3, hyrax, dim);
    proof = hyrax.open(tensor3, eval_point3, c, data_size1, data_size1 / layer_num / input_tokens);
    assert(proof.result == lookup_value);
    normSecondOut_value = normSecondOut_value * SCALE1 + lookup_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");


    data_size1 = load_data("../data/Q/skip-" + to_string(layer_id) + ".bin", &read_tensor1);
    data_size2 = load_data("../data/Q/rmswSecond-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor2.get_data(data_size2, read_tensor2);
    tmp_tensor = tensor1;    //                为了后面的combine claims
    generate_random_eval_points(data_size1, eval_point1);  //skip combine_claims need eval_point1
    tensor3.set_size(data_size2);  tensor3 = tensor2;
    
    auto norm_pair = prove_eleMul(tensor1, tensor2, layer_num, input_tokens, dim, eval_point3, eval_point1, normSecondOut_value);
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");
    //-------------------------------------------lock tmp_tensor----------------------------------
    
    data_size1 = load_data("../data/R/rmswSecond-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor3.set_size(data_size1);  tensor3 = tensor1;
    lookup_value = rs.prove(tensor1, tensor2, eval_point1, hyrax, dim);
    proof = hyrax.open(tensor3, eval_point1, c, data_size1, data_size1 / layer_num / input_tokens);
    assert(proof.result == lookup_value);
    Fr rmsw_value = norm_pair.second * SCALE1 + lookup_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");

    data_size1 = load_data("../data/Q/rmsSecond-" + to_string(layer_id) + ".bin", &read_tensor1);
    data_size2 = load_data("../data/W/NormSecond-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    generate_random_eval_points(layer_num, eval_point3);
    eval_point2 = eval_point1;
    tensor3.get_data(data_size1, read_tensor1);
    tensor2.get_data(data_size2, read_tensor2);
    tensor1.set_size(data_size2);   tensor1 = tensor2;

    auto rmsw_pair = prove_matmul(tensor3, tensor2, layer_num, input_tokens, dim, 1, eval_point2, eval_point3, rmsw_value);
    ReassembleVectors(eval_point2, eval_point3, Log2(input_tokens), Log2(1), Log2(dim));
    proof = hyrax.open(tensor1, eval_point3, c, data_size2, dim / layer_num);
    assert(proof.result == rmsw_pair.second);

    tensor1.get_data(data_size1, read_tensor1);
    proof = hyrax.open(tensor1, eval_point2, c, data_size1, input_tokens / layer_num);   //这两个tensor都是一行的，open函数 N = 单层tensor的长度
    assert(proof.result == rmsw_pair.first);

    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");

    // -------------------------------------------unlock tmp_tensor----------------------------------
    // -------------------------------------------unlock skip_eval_point-----------------------------
    timer.start();
    generate_random_eval_points(dim * layer_num * input_tokens, eval_point2);
    skip_value = combine_claims(tmp_tensor, {skip_value, norm_pair.first}, {skip_eval_point, eval_point1}, 
                                        eval_point2, dim * layer_num * input_tokens);
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");

// ================================norm end===========================================
    
    data_size1 = load_data("../data/Q/input-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    Fr input_value = hyrax.open(tensor1, eval_point2, c, data_size1, data_size1 / layer_num / input_tokens).result;
    Fr atten_out_value = skip_value - input_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");


    data_size1 = load_data("../data/R/atten_out-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor3.set_size(data_size1);  tensor3 = tensor1;
    lookup_value = rs.prove(tensor1, tensor2, eval_point2, hyrax, dim);
    proof = hyrax.open(tensor3, eval_point2, c, data_size1, data_size1 / layer_num / input_tokens);
    assert(proof.result == lookup_value);
    atten_out_value = atten_out_value * SCALE1 + lookup_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");


    data_size1 = load_data("../data/Q/qkv-" + to_string(layer_id) + ".bin", &read_tensor1);
    data_size2 = load_data("../data/W/O-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    generate_random_eval_points(layer_num * dim, eval_point3);

    tensor3.get_data(data_size1, read_tensor1);
    tensor2.get_data(data_size2, read_tensor2);
    tensor1.set_size(data_size2);  tensor1 = tensor2;

    auto O_pair = prove_matmul(tensor3, tensor2, layer_num, input_tokens, dim, dim, eval_point2, eval_point3, atten_out_value);
    ReassembleVectors(eval_point2, eval_point3, Log2(input_tokens), Log2(dim), Log2(dim));
    proof = hyrax.open(tensor1, eval_point3, c, data_size2, dim);
    assert(proof.result == O_pair.second);

    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");


    data_size1 = load_data("../data/R/qkv-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor3.set_size(data_size1);  tensor3 = tensor1;
    lookup_value = rs.prove(tensor1, tensor2, eval_point2, hyrax, input_tokens);
    proof = hyrax.open(tensor3, eval_point2, c, data_size1, data_size1 / layer_num / input_tokens);
    assert(proof.result == lookup_value);
    Fr qkv_out_value = O_pair.first * SCALE1 + lookup_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");


    data_size1 = load_data("../data/Q/softmax_output-" + to_string(layer_id) + ".bin", &read_tensor1);
    data_size2 = load_data("../data/Q/xv-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    generate_random_eval_points(layer_num * input_tokens, eval_point3);

    tensor3.get_data(data_size1, read_tensor1);
    tensor2.get_data(data_size2, read_tensor2);
    tensor1.set_size(data_size1);  tensor1 = tensor3;  //这个要接下来要open matmul的左矩阵

    auto qkv_pair = prove_matmul(tensor3, tensor2, layer_num, input_tokens, dim, input_tokens, eval_point2, eval_point3, qkv_out_value);
    ReassembleVectors(eval_point2, eval_point3, Log2(input_tokens), Log2(input_tokens), Log2(dim));
    skip_eval_point = eval_point3;

    tensor2.set_size(data_size1);  tensor2 = tensor1;
    proof = hyrax.open(tensor1, eval_point2, c, data_size1, input_tokens);
    assert(proof.result == qkv_pair.first);

    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");
    // -------------------------------------------lock skip_eval_point----------------------------------

    data_size1 = load_data("../data/Q/exp_output-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor3.set_size(data_size1);   tensor3 = tensor1;
    tensor3 *= (SCALE1 * 2);
    tensor2 *= 2;
    tensor2 += -1;
    tensor2 *= tensor1.sum(data_size1, input_tokens);
    tensor3 -= tensor2;

    tensor2.set_size(data_size1);
    generate_random_eval_points(data_size1, eval_point1);
    softmax_range_relation.prove(tensor3, tensor2, eval_point1, hyrax, input_tokens);
    
    tensor3.set_size(data_size1); tensor3 = tensor1;
    hyrax.open(tensor1, eval_point1, c, data_size1, input_tokens);

    CUDA_DEBUG;
    total_prove_time += timer.stop("softmax_query");


    data_size1 = load_data("../data/Q/exp_input-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor2.set_size(data_size1);
    generate_random_eval_points(data_size1, eval_point1);
    lookup_value = exp.prove(tensor1, tensor3, tensor2, eval_point1, hyrax, input_tokens);
    proof = hyrax.open(tensor1, eval_point1, c, data_size1, input_tokens);
    Fr exp_in_value = proof.result;
    proof = hyrax.open(tensor3, eval_point1, c, data_size1, input_tokens);
    assert(lookup_value == exp_in_value + proof.result * exp.r);
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");


    data_size1 = load_data("../data/Q/scores-" + to_string(layer_id) + ".bin", &read_tensor1);
    data_size2 = load_data("../data/R/scores-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor2.get_data(data_size2, read_tensor2);
    Fr score_value = tensor1(eval_point1);
    
    tensor3.set_size(data_size1);  tensor3 = tensor2;  
    lookup_value = exp_rs.prove(tensor2, tensor1, eval_point1, hyrax, input_tokens);
    proof = hyrax.open(tensor3, eval_point1, c, data_size1, input_tokens);
    assert(proof.result == lookup_value);
    score_value = score_value * (1 << 22) + lookup_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");


    data_size1 = load_data("../data/Q/xq-" + to_string(layer_id) + ".bin", &read_tensor1);
    data_size2 = load_data("../data/Q/xk-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    generate_random_eval_points(layer_num * dim, eval_point2);

    tensor1.get_data(data_size1, read_tensor1);
    tensor2.get_data(data_size2, read_tensor2);

    auto qk_pair = prove_matmul(tensor1, tensor2, layer_num, input_tokens, input_tokens, dim, eval_point1, eval_point2, score_value);
    ReassembleVectors(eval_point1, eval_point2, Log2(input_tokens), Log2(dim), Log2(input_tokens));
    
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");
    // -------------------------------------------lock eval_point2----------------------------------

    data_size1 = load_data("../data/R/xq-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor3.set_size(data_size1);  tensor3 = tensor1;
    lookup_value = rs.prove(tensor1, tensor2, eval_point1, hyrax, dim);
    proof = hyrax.open(tensor3, eval_point1, c, data_size1, data_size1 / layer_num / input_tokens);
    assert(proof.result == lookup_value);
    Fr q_value = qk_pair.first * SCALE1 + lookup_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");


    data_size1 = load_data("../data/Q/normOutFirst-" + to_string(layer_id) + ".bin", &read_tensor1);
    data_size2 = load_data("../data/W/Q-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    generate_random_eval_points(layer_num * dim, eval_point3);

    tensor3.get_data(data_size1, read_tensor1);
    tensor2.get_data(data_size2, read_tensor2);
    tensor1.set_size(data_size2);  tensor1 = tensor2;

    auto q_pair = prove_matmul(tensor3, tensor2, layer_num, input_tokens, dim, dim, eval_point1, eval_point3, q_value);
    ReassembleVectors(eval_point1, eval_point3, Log2(input_tokens), Log2(dim), Log2(dim));
    proof = hyrax.open(tensor1, eval_point3, c, data_size2, dim);
    assert(proof.result == q_pair.second);
    Fr norm_first_out1 = q_pair.first;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");
    // -------------------------------------------lock eval_point1----------------------------------//先把Q那一支证明完毕
    // -------------------------------------------unlock eval_point2----------------------------------
    data_size1 = load_data("../data/R/xk-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor3.set_size(data_size1);  tensor3 = tensor1;
    lookup_value = rs.prove(tensor1, tensor2, eval_point2, hyrax, dim);
    proof = hyrax.open(tensor3, eval_point2, c, data_size1, data_size1 / layer_num / input_tokens);
    assert(proof.result == lookup_value);
    Fr k_value = qk_pair.second * SCALE1 + lookup_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");


    data_size1 = load_data("../data/Q/normOutFirst-" + to_string(layer_id) + ".bin", &read_tensor1);
    data_size2 = load_data("../data/W/K-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    generate_random_eval_points(layer_num * dim, eval_point3);

    tensor3.get_data(data_size1, read_tensor1);
    tensor2.get_data(data_size2, read_tensor2);
    tensor1.set_size(data_size2);  tensor1 = tensor2;

    auto k_pair = prove_matmul(tensor3, tensor2, layer_num, input_tokens, dim, dim, eval_point2, eval_point3, k_value);
    ReassembleVectors(eval_point2, eval_point3, Log2(input_tokens), Log2(dim), Log2(dim));
    proof = hyrax.open(tensor1, eval_point3, c, data_size2, dim);
    assert(proof.result == k_pair.second);
    Fr norm_first_out2 = k_pair.first;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");
    // -------------------------------------------lock eval_point2----------------------------------//先把Q那一支证明完毕
    // -------------------------------------------unlock skip_eval_point-----------------------------
    data_size1 = load_data("../data/R/xv-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor3.set_size(data_size1);  tensor3 = tensor1;
    lookup_value = rs.prove(tensor1, tensor2, skip_eval_point, hyrax, dim);
    proof = hyrax.open(tensor3, skip_eval_point, c, data_size1, dim);
    assert(proof.result == lookup_value);
    Fr v_value = qkv_pair.second * SCALE1 + lookup_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");


    data_size1 = load_data("../data/Q/normOutFirst-" + to_string(layer_id) + ".bin", &read_tensor1);
    data_size2 = load_data("../data/W/V-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    generate_random_eval_points(layer_num * dim, eval_point3);
    tensor3.get_data(data_size1, read_tensor1);
    tensor2.get_data(data_size2, read_tensor2);
    tensor1.set_size(data_size2);  tensor1 = tensor2;
    // (AB)' = B'A'  证明转置的矩阵乘法
    auto v_pair = prove_matmul(tensor2, tensor3, layer_num, dim, input_tokens, dim, skip_eval_point, eval_point3, v_value);
    ReassembleVectors(skip_eval_point, eval_point3, Log2(dim), Log2(dim), Log2(input_tokens));
    proof = hyrax.open(tensor1, skip_eval_point, c, data_size2, dim);
    assert(proof.result == v_pair.first);
    Fr norm_first_out3 = v_pair.second;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");

    // -------------------------------------------unlock eval_point1 and 2 ----------------------------------
    data_size1 = load_data("../data/Q/normOutFirst-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    generate_random_eval_points(data_size1, skip_eval_point);
    Fr normFirstOut_value = combine_claims(tensor1, {norm_first_out1, norm_first_out2, norm_first_out3}, 
                                {eval_point1, eval_point2, eval_point3}, skip_eval_point, data_size1);
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");

// ================================norm begin===========================================
    eval_point3 = skip_eval_point;
    data_size1 = load_data("../data/R/normOutFirst-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor3.set_size(data_size1);  tensor3 = tensor1;
    lookup_value = rs.prove(tensor1, tensor2, eval_point3, hyrax, dim);
    proof = hyrax.open(tensor3, eval_point3, c, data_size1, data_size1 / layer_num / input_tokens);
    assert(proof.result == lookup_value);
    normFirstOut_value = normFirstOut_value * SCALE1 + lookup_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");


    data_size1 = load_data("../data/Q/input-" + to_string(layer_id) + ".bin", &read_tensor1);
    data_size2 = load_data("../data/Q/rmswFirst-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor2.get_data(data_size2, read_tensor2);
    tmp_tensor = tensor1;    //                为了后面open input
    generate_random_eval_points(data_size1, eval_point1);  //skip combine_claims need eval_point1
    tensor3.set_size(data_size2);  tensor3 = tensor2;
    
    norm_pair = prove_eleMul(tensor1, tensor2, layer_num, input_tokens, dim, eval_point3, eval_point1, normFirstOut_value);
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");
    //-------------------------------------------lock tmp_tensor----------------------------------
    
    data_size1 = load_data("../data/R/rmswFirst-" + to_string(layer_id) + ".bin", &read_tensor1);
    timer.start();
    tensor1.get_data(data_size1, read_tensor1);
    tensor3.set_size(data_size1);  tensor3 = tensor1;
    lookup_value = rs.prove(tensor1, tensor2, eval_point1, hyrax, dim);
    proof = hyrax.open(tensor3, eval_point1, c, data_size1, data_size1 / layer_num / input_tokens);
    assert(proof.result == lookup_value);
    rmsw_value = norm_pair.second * SCALE1 + lookup_value;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");

    data_size1 = load_data("../data/Q/rmsFirst-" + to_string(layer_id) + ".bin", &read_tensor1);
    data_size2 = load_data("../data/W/NormFirst-" + to_string(layer_id) + ".bin", &read_tensor2);
    timer.start();
    generate_random_eval_points(layer_num, eval_point3);
    eval_point2 = eval_point1;
    tensor3.get_data(data_size1, read_tensor1);
    tensor2.get_data(data_size2, read_tensor2);
    tensor1.set_size(data_size2);   tensor1 = tensor2;

    rmsw_pair = prove_matmul(tensor3, tensor2, layer_num, input_tokens, dim, 1, eval_point2, eval_point3, rmsw_value);
    ReassembleVectors(eval_point2, eval_point3, Log2(input_tokens), Log2(1), Log2(dim));
    proof = hyrax.open(tensor1, eval_point3, c, data_size2, dim / layer_num);
    assert(proof.result == rmsw_pair.second);

    tensor1.get_data(data_size1, read_tensor1);
    proof = hyrax.open(tensor1, eval_point2, c, data_size1, input_tokens / layer_num);   //open时 将1 * N，看作4 * (N / 4)
    assert(proof.result == rmsw_pair.first);

    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");
// ================================norm end===========================================
// -------------------------------------------unlock tmp_tensor----------------------------------

    timer.start();
    input_value = hyrax.open(tmp_tensor, eval_point1, c, layer_num * input_tokens * dim, 
                                dim).result;
    CUDA_DEBUG;
    total_prove_time += timer.stop(" ");

}

    cout << "total_prove_time: " << total_prove_time << " S" << endl;


    cudaFree(points);
    cudaFree(read_tensor1);
    cudaFree(read_tensor2);
}