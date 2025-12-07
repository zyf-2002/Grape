
#include <iostream>
#include <iomanip>

#include "g1-tensor.cuh"
#include "ioutils.cuh"

using namespace std;
using namespace alt_bn128;

// Implement G1TensorJacobian

G1TensorJacobian::G1TensorJacobian(const G1TensorJacobian& t)
{
    cudaMalloc((void **)&gpu_data, sizeof(affine_t) * t.size);
    cudaMemcpy(gpu_data, t.gpu_data, sizeof(affine_t) * t.size, cudaMemcpyDeviceToDevice);
}

G1TensorJacobian::G1TensorJacobian(uint size)
{
    cudaMalloc((void **)&gpu_data, sizeof(affine_t) * size);
}

G1TensorJacobian::G1TensorJacobian(uint s, const affine_t* cpu_data)
{
    size = s;
    cudaMalloc((void **)&gpu_data, sizeof(affine_t) * size);
    cudaMemcpy(gpu_data, cpu_data, sizeof(affine_t) * size, cudaMemcpyHostToDevice);
}




G1TensorJacobian::~G1TensorJacobian()
{
   cudaFree(gpu_data);
    gpu_data = nullptr;
}

void G1TensorJacobian::save(const string& filename) const
{
    savebin(filename, gpu_data, size * sizeof(affine_t));
}

G1TensorJacobian::G1TensorJacobian(const string& filename)
{
    cudaMalloc((void **)&gpu_data, size * sizeof(affine_t));
    loadbin(filename, gpu_data, size * sizeof(affine_t));
}

affine_t G1TensorJacobian::operator()(uint idx) const
{
	affine_t out;
	cudaMemcpy(&out, gpu_data + idx, sizeof(affine_t), cudaMemcpyDeviceToHost);
	return out;
}

