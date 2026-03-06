// #include "rescaling.cuh"
// #include "ioutils.cuh"  // 包含 random_vec 函数

// // 构造函数定义
// Rescaling::Rescaling(uint scaling_factor) 
//     : scaling_factor(scaling_factor), 
//       tl_rem(-static_cast<int>(scaling_factor>>1), scaling_factor)
// {
// }

// // prove 方法定义
// Fr Rescaling::prove(FrTensor& X, FrTensor &A, const vector<Fr>& v, Hyrax &hyrax, uint N)  //N是Hyrax承诺A的每行的大小
// {
//     auto u = random_vec(Log2(X.size)); 
//     assert(u.size() == v.size());
//     uint size = X.size;
//     tl_rem.prep(X);
    
//     if(size % 11008 == 0){
//         X.pad(A, 11008, 16384, size / 11008 * 16384, Fr::zero());
//     }

//     Fr claim = tl_rem.prove(X, A, u, v, hyrax, N);
//     X.set_size(size);                         //将大小设置为原始大小
//     CUDA_DEBUG;
//     std::cout << "Rescaling proof complete." << std::endl;
//     return claim;
// }