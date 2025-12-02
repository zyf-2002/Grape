#include <iostream>
#include <iomanip>
#include "./field/alt_bn128.cuh"
using namespace std;
using namespace alt_bn128;

class G1TensorJacobian
{
    public: 
    
    uint size;

    affine_t* gpu_data;

    G1TensorJacobian(const G1TensorJacobian&);

    G1TensorJacobian(uint size);

    
    G1TensorJacobian(uint size, const affine_t* cpu_data);


    G1TensorJacobian(const string& filename);

    ~G1TensorJacobian();

    void save(const string& filename) const;

	affine_t operator()(uint) const;

    G1TensorJacobian operator-() const;

    
};

