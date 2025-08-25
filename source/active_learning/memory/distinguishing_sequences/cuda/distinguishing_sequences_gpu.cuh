/**
 * @file distinguishing_sequences_gpu.cuh
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
static_assert(std::integral_constant<bool, false>::value, "distinguishing_sequences_gpu.cuh included even though CUDA not enabled in project.");
#endif

#ifndef __KERNELS_DISTINGUISHING_SEQUENCES_CUH__
#define __KERNELS_DISTINGUISHING_SEQUENCES_CUH__

namespace distinguishing_sequences_gpu {
  float get_overlap_gpu(const int* d1_d, const int* d2_d, const size_t size);
}

#endif // __KERNELS_DISTINGUISHING_SEQUENCES_CUH__