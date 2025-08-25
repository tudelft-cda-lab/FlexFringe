/**
 * @file nn_discrete_and_float_output.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Used e.g. when we have a discrete output, but each is accompanied by a floating point output as well. 
 * The discrete output could represent a classification value, and the floating point the cofidence of the network.
 * @version 0.1
 * @date 2024-11-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _NN_DISCRETE_AND_FLOAT_OUTPUT_SUL_H_
#define _NN_DISCRETE_AND_FLOAT_OUTPUT_SUL_H_

#include "nn_sul_base.h"

#ifdef __FLEXFRINGE_PYTHON

#if defined(_MSC_VER) 
#  define FLEXFRINGE_ALWAYS_INLINE inline
#else
#  define FLEXFRINGE_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

/**
 * @brief Used e.g. when we have a discrete output, but each is accompanied by a floating point output as well. 
 * The discrete output could represent a classification value, and the floating point the cofidence of the network.
 * 
 */
class nn_discrete_and_float_output_sul : public nn_sul_base {
  public:
    void reset() override {};

    const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) const override;
    const sul_response do_query(const std::vector< std::vector<int> >& query_traces, inputdata& id) const override;
};

#undef FLEXFRINGE_ALWAYS_INLINE

#else

class nn_discrete_and_float_output_sul : public nn_sul_base {
  public:
    void reset() override {};

    const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) const override;
    const sul_response do_query(const std::vector< std::vector<int> >& query_traces, inputdata& id) const override;
};

#endif // __FLEXFRINGE_PYTHON
#endif // _NN_DISCRETE_AND_FLOAT_OUTPUT_SUL_H_