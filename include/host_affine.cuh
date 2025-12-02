#include <iostream>

#include "libff/common/profiling.hpp"
#include "libsnark/common/default_types/r1cs_ppzksnark_pp.hpp"
#include "libff/algebra/scalar_multiplication/multiexp.hpp"

#include <omp.h>

using namespace std;


using namespace libsnark;
typedef default_r1cs_ppzksnark_pp ppT;



class G1_affine {
public:
    
    libff::alt_bn128_Fq X;
    libff::alt_bn128_Fq Y;

    
    G1_affine() {}
    G1_affine(libff::alt_bn128_G1 g) : X(g.X), Y(g.Y) {}
};