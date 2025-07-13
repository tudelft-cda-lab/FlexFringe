/**
 * @file cuda_common.cuh
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-07-12
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef __FLEXFRINGE_CUDA
#include<type_traits>
static_assert(std::integral_constant<bool, false>::value, "cuda_common.cuh included even though CUDA not enabled in project.");
#endif

#ifndef __CUDA_COMMON_CUH__
#define __CUDA_COMMON_CUH__

#include "cuda.h"

namespace cuda_common {  
  void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);
}

#endif // __CUDA_COMMON_CUH__