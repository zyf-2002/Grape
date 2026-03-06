// #pragma once
// #include <thrust/device_vector.h>
// #include <thrust/sequence.h>

// #include "./field/alt_bn128.cuh"
// #include "./fr-tensor.cuh"
// #include "./ioutils.cuh"
// #include "ec_operation.cuh"
// #include "polynomial.cuh"
// #include <cub/cub.cuh>
// using namespace alt_bn128;

// __global__ void get_rs_kernel(const uint* idx, uint* RS_count, const uint* prefix_RS_count, const uint* S_count) {
//         uint gid = GET_GLOBAL_ID();
//         if(gid == 32768) return;
//         uint count = 0;
//         uint start_pos = prefix_RS_count[gid];
//         uint end_pos = prefix_RS_count[gid] + S_count[gid];
//         for(uint i = start_pos; i < end_pos; i++){
//             RS_count[idx[i]] = count;
//             count++;
//         }
// }
// __global__ void get_rs_zero_kernel(const uint* idx, uint* RS_count, uint size) {
//         uint gid = GET_GLOBAL_ID();
//         uint start = gid * 256;
//         if(start >= size) return;
//         uint count = gid * 256;
//         uint end = min(start + 256, size);
//         for(int i = start; i < end; i++){
//             RS_count[idx[i]] = count;
//             count++;
//         }
// }



// __global__ void get_dim_kernel(const int* query, uint* dim, const uint size, const int lower_bound) {
//         uint gid = GET_GLOBAL_ID();
//         if(gid < size){
//             dim[gid] = query[gid] - lower_bound;
//         }
// }

// __global__ void build_hist(const int* input, uint* hist, int N, int lower_bound) {
//     int i = GET_GLOBAL_ID();
//     if (i < N) {
//         atomicAdd(&hist[input[i] - lower_bound], 1);
//     }
// }

// Fr WS_init(const int Lower_bound, const int Upper_bound){
//     Fr WS = Fr::one();
//     Fr lower_bound(Lower_bound);
//     Fr index = Fr::zero();
//     for(int i = Lower_bound; i <= Upper_bound; i++){
//         WS *= (index + lower_bound);
//         lower_bound = lower_bound + Fr::one();
//         index = index + Fr::one();
//     }
//     return WS;
// }

// void lasso_init(const int *query, uint *dim, uint *RS_count, uint *S_count, const int size, const int Lower_bound){
//     CUDA_DEBUG;
    
//     uint *idx = nullptr;
//     int* query_tmp = nullptr;   
//     uint* idx_tmp  = nullptr;   // 排序后的 index
//     void* temp    = nullptr;   //临时计算空间
//     uint* S_prefix;
//     size_t temp_bytes_sort = 0;
//     size_t temp_bytes_prefixsum = 0;

//     cudaMalloc(&S_prefix, 65536 * sizeof(uint));
//     cudaMalloc(&query_tmp, size * sizeof(int));   // 必须，但不用
//     cudaMalloc(&idx_tmp,  size * sizeof(uint));
//     cudaMalloc(&idx, size * sizeof(uint));

//     thrust::sequence(thrust::device, idx, idx + size);
//     // 3. 查询临时空间
//     cub::DeviceRadixSort::SortPairs(
//         nullptr, temp_bytes_sort,
//         query, query_tmp,
//         idx, idx_tmp,
//         size
//     );
//     cub::DeviceScan::ExclusiveSum(
//         nullptr, temp_bytes_prefixsum,
//         S_count, S_prefix,
//         65536
//     );
//     size_t temp_bytes = max(temp_bytes_sort, temp_bytes_prefixsum);

//     cudaMalloc(&temp, temp_bytes);
//     // 4. 真正排序
//     cub::DeviceRadixSort::SortPairs(
//         temp, temp_bytes,
//         query, query_tmp,
//         idx, idx_tmp,
//         size
//     );

//     cudaMemcpy(idx, idx_tmp, size * sizeof(uint), cudaMemcpyDeviceToDevice);

//     cub::DeviceScan::ExclusiveSum(
//         temp, temp_bytes,
//         S_count, S_prefix,
//         65536
//     );

//     get_rs_kernel<<<65536 / 256, 256>>>(idx, RS_count, S_prefix, S_count); 
//     uint zero_count = 0;
//     cudaMemcpy(&zero_count, S_count + (-Lower_bound), sizeof(uint), cudaMemcpyDeviceToHost);
//     assert(zero_count >= size - 1024 * 4 * 11008);
//     uint thread = (zero_count + 255) / 256;
//     uint zero_prefix;
//     cudaMemcpy(&zero_prefix, S_prefix + (-Lower_bound), sizeof(uint), cudaMemcpyDeviceToHost);
//     get_rs_zero_kernel<<<(thread + 255) / 256, 256>>>(idx + zero_prefix, RS_count, zero_count);

//     // uint *rs_host = new uint[size];
//     // cudaMemcpy(rs_host, RS_count, size * sizeof(uint), cudaMemcpyDeviceToHost);
//     // uint *RS_host= new uint[size];
//     // int *qh = new int[size];
//     // cudaMemcpy(qh, query, size * sizeof(int), cudaMemcpyDeviceToHost);
//     // uint *ct = new uint[65536];
//     // memset(ct, 0, 65536 * sizeof(uint));
//     // for(int i = 0; i < size; i++){
//     //     RS_host[i] = ct[qh[i] - Lower_bound];
//     //     ct[qh[i] - Lower_bound]++;
//     // }
//     // for(int i = 0; i < size; i++){
//     //     assert(RS_host[i] == rs_host[i]);
//     // }
   
//     get_dim_kernel<<<(size + 255) / 256, 256>>>(query, dim, size, Lower_bound);
//     cudaFree(idx);
//     cudaFree(query_tmp);
//     cudaFree(idx_tmp);
//     cudaFree(temp);
//     cudaFree(S_prefix);
// }