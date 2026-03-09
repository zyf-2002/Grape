#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include "fr-tensor.cuh"
#include "Hyrax.cuh"
#include "ec_operation.cuh"
#include "timer.cuh"
#include <omp.h>

using namespace std;
using namespace libsnark;

uint npoints = 16384;
uint W = 1 << 8;  //pre_windows
uint input_tokens = 2048;
uint layer_num = 1;

struct FileInfo {
    std::string relative_path;  
    uint N;
};

int main(int argc, char* argv[])
{
    ppT::init_public_params();
    affine_t *points;
    cudaMalloc((void **)&points, sizeof(affine_t) * npoints * W * layer_num);
    auto cpu_points = precompute_generators(npoints * layer_num, W, points);

    savebin("../data/points.bin", points, sizeof(affine_t) * npoints * W * layer_num);
    savebin("../data/cpu_points.bin", cpu_points.data(), cpu_points.size() * sizeof(bn128), false);
    CUDA_DEBUG;

    Hyrax hyrax(layer_num, npoints, points, cpu_points[0]);

    cout << "---------------------------------commit W---------------------------------" << endl;
    CUDA_DEBUG;
    Timer timer;
    std::vector<FileInfo> W_files = {
        {"down-", 11008},
        {"up-", 4096},
        {"gate-", 4096},
        {"NormSecond-", 4096 / layer_num},
        {"NormFirst-", 4096 / layer_num},
        {"O-", 4096},
        {"K-", 4096},
        {"Q-", 4096},
        {"V-", 4096}
    };
    
    double w_commit_time = 0;
    for (const auto& file : W_files) {
        for(int l = (32 / layer_num) - 1; l >= 0; l--){
            int *tensor = nullptr;

            string filename = "../data/W/" + file.relative_path + to_string(l) + ".bin";
            uint size = load_data(filename, &tensor); 
            timer.start();
            uint N = file.N;
            pad_int(&tensor, N, 1 << Log2(N), (size/N)*(1<<Log2(N)));
            jacob_t *commitment = hyrax.commit(tensor, (size/N)*(1<<Log2(N)), 1 << Log2(N));
            CUDA_DEBUG;
            w_commit_time += timer.stop("commit_w");

            string com_filename = "../data/comW/" + file.relative_path;
            savebin(com_filename, commitment, sizeof(jacob_t) * (size / N / layer_num));
            CUDA_DEBUG;
            cout << "Processed: " << filename 
                << ", N = " << N 
                << ", commitment saved." << endl;

            cudaFree(commitment);
            cudaFree(tensor);
        }              
        
    }

    cout << "---------------------------------commit R---------------------------------" << endl;

    std::vector<FileInfo> R_files = {
        {"down_out-", 4096},
        {"up_out-", 11008},
        {"gate_out-", 11008},
        {"upSilu-", 11008},
        {"atten_out-", 4096},
        {"normOutFirst-", 4096},
        {"normOutSecond-", 4096},
        {"qkv-", 4096},
        {"rmswFirst-", 4096},
        {"rmswSecond-", 4096},
        {"scores-", input_tokens},
        {"xk-", 4096},
        {"xv-", 4096},
        {"xq-", 4096}
    };

    double r_commit_time = 0;
    for (const auto& file : R_files) {
        for(int l = (32 / layer_num) - 1; l >= 0; l--){
            int *tensor = nullptr;
        
            string filename = "../data/R/" + file.relative_path + to_string(l) + ".bin";
            uint size = load_data(filename, &tensor);
        
            timer.start();
            uint N = file.N;
            pad_int(&tensor, N, 1 << Log2(N), (size/N)*(1<<Log2(N)));
            jacob_t *commitment = hyrax.commit(tensor, (size/N)*(1<<Log2(N)), 1 << Log2(N));
            CUDA_DEBUG;
            r_commit_time += timer.stop("commit_r");

            string com_filename = "../data/comR/" + file.relative_path;
            savebin(com_filename, commitment, sizeof(jacob_t) * (size / N / layer_num));
            CUDA_DEBUG;
            cout << "Processed: " << filename 
                << ", N = " << N 
                << ", commitment saved." << endl;

            cudaFree(commitment);
            cudaFree(tensor);
        }
    }

    cout << "---------------------------------commit Q---------------------------------" << endl;

    std::vector<FileInfo> Q_files = {
        {"exp_input-", input_tokens},
        {"exp_output-", input_tokens},
        {"softmax_output-", input_tokens},
        {"rmsFirst-", input_tokens / layer_num},
        {"rmsSecond-", input_tokens / layer_num},
        {"gate_out-", 11008},
        {"silu_out-", 11008},
        {"input-", 4096},
        {"output-", 4096}
    };

    double q_commit_time = 0;
    for (const auto& file : Q_files) {
        for(int l = (32 / layer_num) - 1; l >= 0; l--){
            int *tensor = nullptr;
        
            string filename = "../data/Q/" + file.relative_path + to_string(l) + ".bin";
            uint size = load_data(filename, &tensor);
        
            timer.start();
            uint N = file.N;
            pad_int(&tensor, N, 1 << Log2(N), (size/N)*(1<<Log2(N)));
            jacob_t *commitment = hyrax.commit(tensor, (size/N)*(1<<Log2(N)), 1 << Log2(N));
            CUDA_DEBUG;
            q_commit_time += timer.stop("commit_q");    

            string com_filename = "../data/comQ/" + file.relative_path;
            savebin(com_filename, commitment, sizeof(jacob_t) * (size / N / layer_num));
            CUDA_DEBUG;
            cout << "Processed: " << filename 
                << ", N = " << N 
                << ", commitment saved." << endl;

            cudaFree(commitment);
            cudaFree(tensor);
        }
        
    }

    cout << "---------------------------------commit done---------------------------------" << endl;

    cout << "W Commit time: " << w_commit_time << " ms" << endl;
    cout << "R Commit time: " << r_commit_time << " ms" << endl;
    cout << "Q Commit time: " << q_commit_time << " ms" << endl;
    return 0;
}