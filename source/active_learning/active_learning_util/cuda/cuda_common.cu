/**
 * @file cuda_common.cu
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
static_assert(std::integral_constant<bool, false>::value, "cuda_common.cu included even though CUDA not enabled in project.");
#endif

#include "cuda_common.cuh"

#include <iostream>

void cuda_common::gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
  if (code != cudaSuccess) 
  {
      //fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line;
      if (abort) exit(code);
  }
}