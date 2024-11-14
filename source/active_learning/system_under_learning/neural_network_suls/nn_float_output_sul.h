/**
 * @file nn_float_output_sul.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Self explanatory.
 * @version 0.1
 * @date 2024-11-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _NN_FLOAT_OUTPUT_SUL_H_
#define _NN_FLOAT_OUTPUT_SUL_H_

#include "nn_sul_base.h"

#if defined(_MSC_VER) 
#  define FLEXFRINGE_ALWAYS_INLINE inline
#else
#  define FLEXFRINGE_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

/**
 * @brief When we expect flaoting point return values from the network. E.g. when they return probabilities.
 * 
 */
class nn_float_output_sul : public nn_sul_base {
  protected:
    void reset() override {};

    const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) const override;
    const sul_response do_query(const std::vector< std::vector<int> >& query_traces, inputdata& id) const override;

  public:
    nn_float_output_sul() : nn_sul_base() {};
};

#undef FLEXFRINGE_ALWAYS_INLINE

#endif // _nn_float_output_sul_H_