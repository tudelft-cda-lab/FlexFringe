/**
 * @file distinguishing_sequences_kernels.cu
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-07-13
 * 
 * @copyright Copyright (c) 2025
 * 
 */

 #ifndef __FLEXFRINGE_CUDA
#include<type_traits>
static_assert(std::integral_constant<bool, false>::value, "distinguishing_sequences_gpu.cu included even though CUDA not enabled in project.");
#endif

#include "distinguishing_sequences_gpu.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_common.cuh"

#ifndef gpuErrcheck
#define gpuErrchk(ans) { cuda_common::gpuAssert((ans), __FILE__, __LINE__); }


 // unnamed namespace only accessible in file of definition
namespace {
  /**
   * @brief Example of how to use below in this file.
   */

   // fix here? https://forums.developer.nvidia.com/t/cmake-compile-cpp-as-cu/204118
  __global__ void sum_kernel(int* input, int* res, const int size){
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid>=size)
      return;

    // local data block pointer
    int* i_data = input + blockDim.x * blockIdx.x;
    for(int offset=1; offset <= blockDim.x; offset *= 2){
      int index = 2*offset*tid;

      if(index < blockDim.x){
        i_data[index] += i_data[index+offset];
      }

      __syncthreads();
    }

    if(tid==0){
      res[blockIdx.x] = input[gid];
    }
  }

    /**
   * @brief "XORs" two vectors. Result set to one if entry in v1 and v2 same, else 0.
   */
  __global__ void xor_vectors_kernel(const int* v1, const int* v2, int* tmp, const int size){
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    if(gid >= size)
      return;

    tmp[gid] = v1[gid] == v2[gid] ? 1 : 0;
  }
}

  /**
   * @brief Get the overlap in percent of two int-vectors of given size.
   * Useful for e.g. computing the accuracy score in between two vectors.
   * 
   * ..._d indicates a data structure on device- (gpu-) memory. 
   * _...h for host ("cpu-visible" memory)
   * 
   * TODO: inefficiency by allocating/recallocating memory
   */
  float distinguishing_sequences_gpu::get_overlap_gpu(const int* d1_d, const int* d2_d, const size_t size){
    dim3 block(256);
    dim3 grid(size/block.x);

    int* res_arr_d;
    const auto res_arr_byte_size = grid.x * sizeof(int);
    gpuErrchk( cudaMalloc((void**) &res_arr_d, res_arr_byte_size) );
    gpuErrchk( cudaMemset(res_arr_d, 0, res_arr_byte_size) );

    int* tmp_arr_d;
    gpuErrchk( cudaMalloc((void**) &tmp_arr_d, size * sizeof(int)) );

    xor_vectors_kernel<<<grid, block>>>(d1_d, d2_d, tmp_arr_d, size);
    gpuErrchk( cudaDeviceSynchronize() );

    sum_kernel<<<grid, block>>>(tmp_arr_d, res_arr_d, size);
    gpuErrchk( cudaDeviceSynchronize() );

    int* res_arr_h;
    res_arr_h = (int*) malloc(res_arr_byte_size);
    cudaMemcpy(res_arr_h, res_arr_d, res_arr_byte_size, cudaMemcpyDeviceToHost);

    int n_overlaps = 0;
    for(int i = 0; i < grid.x; ++i){
      n_overlaps += res_arr_h[i];
    }

    free(res_arr_h);
    gpuErrchk( cudaFree(tmp_arr_d) );
    gpuErrchk( cudaFree(res_arr_d) );

    return static_cast<float>(n_overlaps) / static_cast<float>(size);
}

#endif // gpuErrcheck