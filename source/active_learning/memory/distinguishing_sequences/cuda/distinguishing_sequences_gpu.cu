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
    //int* i_data = input + blockDim.x * blockIdx.x;
    for(int offset=1; offset <= blockDim.x/2; offset *= 2){
      /* int index = 2*offset*tid;

      if(index < blockDim.x){
        i_data[index] += i_data[index+offset];
      } */

      if(tid % (2 * offset) == 0){
        input[gid] += input[gid + offset];
      }

      __syncthreads();
    }

    if(tid==0){
      res[blockIdx.x] = input[gid];
    }
  }

  // TODO: debug
  __global__ void sum_kernel_unrolled(int* input, int* res, const int size){
    int tid = threadIdx.x;

    // local data pointer
    int* i_data = input + blockDim.x * blockIdx.x;

    if(blockDim.x == 1024 && tid < 512){
      i_data[tid] += i_data[tid+512];
    }
    __syncthreads();

    if(blockDim.x == 512 && tid < 256){
      i_data[tid] += i_data[tid+256];
    }
    __syncthreads();

    if(blockDim.x == 256 && tid < 128){
      i_data[tid] += i_data[tid+128];
    }
    __syncthreads();

    if(blockDim.x == 128 && tid < 64){
      i_data[tid] += i_data[tid+64];
    }
    __syncthreads();

    // a number lower than 32 would introduce warp divergence, but like this it is fine
    if(tid < 32){\
      // volatile here guarantees that memory load and store to global memory without any caches
      volatile int* vsmem = i_data;

      // here we unroll the loop
      vsmem[tid] += vsmem[tid+32];
      vsmem[tid] += vsmem[tid+16];
      vsmem[tid] += vsmem[tid+8];
      vsmem[tid] += vsmem[tid+4];
      vsmem[tid] += vsmem[tid+2];
      vsmem[tid] += vsmem[tid+1];
    }

    if(tid==0){
      res[blockIdx.x] = i_data[0];
    }
  }

    /**
   * @brief "XORs" two vectors. Result set to one if entry in v1 and v2 same, else 0.
   */
  __global__ void xor_vectors_kernel(const int* v1, const int* v2, int* tmp, const int size){
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
    const int threads_per_block = 256;
    const int n_blocks = max(1, static_cast<int>(size/threads_per_block));

    dim3 block(threads_per_block);
    dim3 grid(n_blocks);

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

    constexpr static float epsilon = 1e-6; // avoid division error when v1 or v2 only have -1 entries, or size of this is 0
    return static_cast<float>(n_overlaps) / (static_cast<float>(size) + epsilon);
}

#endif // gpuErrcheck