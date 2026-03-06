// #pragma once
// #include "fr-tensor.cuh" 
// #include "tlookup.cuh"
// #include <vector>
// #include <cassert>
// #include <iostream>
// #include <omp.h>

// class Rescaling {
// public:
//     const uint scaling_factor;
//     tLookupRange tl_rem; // table for remainder
    
//     Rescaling(uint scaling_factor);
//     Fr prove(FrTensor& X, FrTensor &A, const vector<Fr>& v, Hyrax &hyrax, uint N); //N是Hyrax承诺A的每行的大小
// };