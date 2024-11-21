/**
 * @file nn_float_vector_output_sul.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief When we expect vector of floating point numbers from the network, such as probability distributions.
 * @version 0.1
 * @date 2024-11-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _NN_FLOAT_VECTOR_OUTPUT_SUL_H_
#define _NN_FLOAT_VECTOR_OUTPUT_SUL_H_

#include "nn_sul_base.h"

#if defined(_MSC_VER) 
#  define FLEXFRINGE_ALWAYS_INLINE inline
#else
#  define FLEXFRINGE_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

/**
 * @brief When we expect vector of floating point numbers from the network. 
 * This can be for example probability distributions, used to build PDFAs.
 * 
 */
class nn_float_vector_output_sul : public nn_sul_base {
  protected:
    void reset() override {};
    const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) const override;

  public:
    nn_float_vector_output_sul(const std::string& cf) : nn_sul_base(cf) {};
};

#undef FLEXFRINGE_ALWAYS_INLINE

#endif // _NN_FLOAT_VECTOR_OUTPUT_SUL_H_