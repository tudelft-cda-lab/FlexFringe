/**
 * @file nn_binary_output_sul.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This SUL is used e.g. when the underlying network has a sigmoid output. Returns the zero- or one-values directly.
 * @version 0.1
 * @date 2024-11-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _NN_BINARY_OUTPUT_SUL_H_
#define _NN_BINARY_OUTPUT_SUL_H_

#include "nn_sul_base.h"

#if defined(_MSC_VER) 
#  define FLEXFRINGE_ALWAYS_INLINE inline
#else
#  define FLEXFRINGE_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

/**
 * @brief Used e.g. when the underlying network has a sigmoid output. Returns the zero- or one-values directly.
 * 
 */
class nn_binary_output_sul : public nn_sul_base {
  protected:
    void reset() override {};

#ifdef __FLEXFRINGE_PYTHON
    FLEXFRINGE_ALWAYS_INLINE const double get_sigmoid_output(const std::vector<int>& query_trace, inputdata& id) const;
#endif // __FLEXFRINGE_PYTHON

    const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) const override;

  public:
    nn_binary_output_sul(const std::string& cf) : nn_sul_base(cf){};
};

#undef FLEXFRINGE_ALWAYS_INLINE
#endif
