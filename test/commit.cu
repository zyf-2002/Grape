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

uint npoints = 11008;
const size_t W = 1 << 8;  //pre_windows
uint input_tokens = 2048;

struct FileInfo {
    std::string relative_path;  
    uint N;
};

int main(int argc, char* argv[])
{
    ppT::init_public_params();
    affine_t *points;
    cudaMalloc((void **)&points, sizeof(affine_t) * npoints * W);
    auto cpu_points = precompute_generators(npoints, W, points);

    savebin("../data/points.bin", points, sizeof(affine_t) * npoints * W);
    savebin("../data/cpu_points.bin", cpu_points.data(), cpu_points.size() * sizeof(bn128), false);
    CUDA_DEBUG;

    Hyrax hyrax(npoints, points, cpu_points[0]);

    cout << "---------------------------------commit W---------------------------------" << endl;
    Timer timer;
    std::vector<FileInfo> W_files = {
        {"down-7.bin", 11008},
        {"up-7.bin", 11008},
        {"gate-7.bin", 11008},
        {"NormSecond-7.bin", 4096},
        {"NormFirst-7.bin", 4096},
        {"O-7.bin", 4096},
        {"K-7.bin", 4096},
        {"Q-7.bin", 4096},
        {"V-7.bin", 4096},
    };

    double w_commit_time = 0;
    for (const auto& file : W_files) {
        int *tensor = nullptr;
        
        string filename = "../data/W/" + file.relative_path;
        uint size = load_data(filename, &tensor);
        
        timer.start();
        uint N = file.N;
        jacob_t *commitment = hyrax.commit(tensor, size, N);
        CUDA_DEBUG;
        w_commit_time += timer.stop("commit_w");

        string com_filename = "../data/comW/" + file.relative_path;
        savebin(com_filename, commitment, sizeof(jacob_t) * (size / N));
        CUDA_DEBUG;
        cout << "Processed: " << filename 
             << ", N = " << N 
             << ", commitment saved." << endl;

        cudaFree(commitment);
        cudaFree(tensor);
    }

    cout << "---------------------------------commit R---------------------------------" << endl;

    std::vector<FileInfo> R_files = {
        {"down_out-7.bin", 11008},
        {"up_out-7.bin", 11008},
        {"gate_out-7.bin", 11008},
        {"upSilu-7.bin", 11008},
        {"atten_out-7.bin", 4096},
        {"normOutFirst-7.bin", 4096},
        {"normOutSecond-7.bin", 4096},
        {"qkv-7.bin", 4096},
        {"rmswFirst-7.bin", 4096},
        {"rmswSecond-7.bin", 4096},
        {"scores-7.bin", input_tokens},
        {"xk-7.bin", 4096},
        {"xv-7.bin", 4096},
        {"xq-7.bin", 4096}
    };

    double r_commit_time = 0;
    for (const auto& file : R_files) {
        int *tensor = nullptr;
        
        string filename = "../data/R/" + file.relative_path;
        uint size = load_data(filename, &tensor);
        
        timer.start();
        uint N = file.N;
        jacob_t *commitment = hyrax.commit(tensor, size, N);
        CUDA_DEBUG;
        r_commit_time += timer.stop("commit_r");

        string com_filename = "../data/comR/" + file.relative_path;
        savebin(com_filename, commitment, sizeof(jacob_t) * (size / N));
        CUDA_DEBUG;
        cout << "Processed: " << filename 
             << ", N = " << N 
             << ", commitment saved." << endl;

        cudaFree(commitment);
        cudaFree(tensor);
    }

    cout << "---------------------------------commit Q---------------------------------" << endl;

    std::vector<FileInfo> Q_files = {
        {"exp_input-7.bin", input_tokens},
        {"exp_output-7.bin", input_tokens},
        {"rmsFirst-7.bin", input_tokens},
        {"rmsSecond-7.bin", input_tokens},
        {"gate_out-7.bin", 11008},
        {"silu_out-7.bin", 11008},
        {"input-7.bin", 4096},
        {"output-7.bin", 4096}
    };

    double q_commit_time = 0;
    for (const auto& file : Q_files) {
        int *tensor = nullptr;
        
        string filename = "../data/Q/" + file.relative_path;
        uint size = load_data(filename, &tensor);
        
        timer.start();
        uint N = file.N;
        jacob_t *commitment = hyrax.commit(tensor, size, N);
        CUDA_DEBUG;
        q_commit_time += timer.stop("commit_q");    

        string com_filename = "../data/comQ/" + file.relative_path;
        savebin(com_filename, commitment, sizeof(jacob_t) * (size / N));
        CUDA_DEBUG;
        cout << "Processed: " << filename 
             << ", N = " << N 
             << ", commitment saved." << endl;

        cudaFree(commitment);
        cudaFree(tensor);
    }

    cout << "---------------------------------commit done---------------------------------" << endl;

    cout << "W Commit time: " << w_commit_time << " ms" << endl;
    cout << "R Commit time: " << r_commit_time << " ms" << endl;
    cout << "Q Commit time: " << q_commit_time << " ms" << endl;
    return 0;
}