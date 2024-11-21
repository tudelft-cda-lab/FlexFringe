/**
 * @file nn_discrete_and_float_output.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Used when we expect a discrete output, but also collect the hidden representation in the form
 * of a vector of floating point values.
 * 
 * IMPORTANT: This class expects a causal network. The reason being is that we currently only support 
 * the hidden representation of the last symbol. In a non-causal model, the hidden representation 
 * can be changed by future values, but in a non-causal model they cannot. Hence, in a non-causal model 
 * all hidden representations have to be analyzed, but in a causal model it is enough to look at the last 
 * symbol at a time, assuming that all prefixes of the current string are known.
 * 
 * @version 0.1
 * @date 2024-11-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _NN_DISCRETE_OUTPUT_AND_HIDDEN_REPS_SUL_H_
#define _NN_DISCRETE_OUTPUT_AND_HIDDEN_REPS_SUL_H_

#include "nn_sul_base.h"

#ifdef __FLEXFRINGE_PYTHON

#if defined(_MSC_VER) 
#  define FLEXFRINGE_ALWAYS_INLINE inline
#else
#  define FLEXFRINGE_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

/**
 * @brief Used when we expect a discrete output, but also collect the hidden representation in the form
 * of a vector of floating point values.
 * 
 * IMPORTANT: This class expects a causal network. The reason being is that we currently only support 
 * the hidden representation of the last symbol. In a non-causal model, the hidden representation 
 * can be changed by future values, but in a non-causal model they cannot. Hence, in a non-causal model 
 * all hidden representations have to be analyzed, but in a causal model it is enough to look at the last 
 * symbol at a time, assuming that all prefixes of the current string are known.
 * 
 * We implemented a function retrieving all hidden representations and an example of an efficient implementation
 * in the accompanying .cpp file of this header (overloaded nn_discrete_output_and_hidden_reps_sul()). The function 
 * is unused however and will NOT LINK, unless you use it somewhere.
 */
class nn_discrete_output_and_hidden_reps_sul : public nn_sul_base {
  private:
      FLEXFRINGE_ALWAYS_INLINE std::vector<float> compile_hidden_rep(PyObject* p_result, const int offset) const;

  protected:
    void reset() override {};

  public:
    nn_discrete_output_and_hidden_reps_sul() : nn_sul_base() {};
    const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) const override;
};

#undef FLEXFRINGE_ALWAYS_INLINE

#else 

class nn_discrete_output_and_hidden_reps_sul : public nn_sul_base {
  protected:
    const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) const override;

  public:
    nn_discrete_output_and_hidden_reps_sul(const std::string& cf) : nn_sul_base(cf){};
};

#endif // __FLEXFRINGE_PYTHON
#endif // _NN_DISCRETE_OUTPUT_AND_HIDDEN_REPS_SUL_H_