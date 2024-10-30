/**
 * @file nn_weighted_output_sul.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-11-29
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _NN_WEIGHTED_OUTPUT_SUL_H_
#define _NN_WEIGHTED_OUTPUT_SUL_H_

#include "nn_sul_base.h"

#if defined(_MSC_VER) 
#  define FLEXFRINGE_ALWAYS_INLINE inline
#else
#  define FLEXFRINGE_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

class nn_weighted_output_sul : public nn_sul_base {
    friend class base_teacher;
    friend class eq_oracle_base;

  private:
    FLEXFRINGE_ALWAYS_INLINE const double get_sigmoid_output(const std::vector<int>& query_trace,
                                                                          inputdata& id) const;
#ifdef __FLEXFRINGE_PYTHON
    inline std::vector< std::vector<float> > compile_hidden_rep(PyObject* p_result, const int offset) const;
#endif /* __FLEXFRINGE_PYTHON */

  protected:
    void reset() override {};
    void init_types() const override;

#ifdef __FLEXFRINGE_PYTHON
    void string_to_pylist(PyObject* p_list_out, std::vector<std::string>& c_list) const;
#endif /* __FLEXFRINGE_PYTHON */

    /* Learning from NN acceptors*/
    bool is_member(const std::vector<int>& query_trace) const;
    const int query_trace(const std::vector<int>& query_trace, inputdata& id) const override;
    const std::pair< int, std::vector< std::vector<float> > > get_type_and_states(const std::vector<int>& query_trace, inputdata& id) const override;
    const std::tuple< int, float, std::vector< std::vector<float> > > get_type_confidence_and_states(const std::vector<int>& query_trace, inputdata& id) const override;
    const std::vector< std::pair<int, float> > get_type_confidence_batch(const std::vector< std::vector<int> >& query_traces, inputdata& id) const override;

    /* Learning from Language Models */
    const double get_string_probability(const std::vector<int>& query_trace, inputdata& id) const override;
    const std::vector<float> get_weight_distribution(const std::vector<int>& query_trace,
                                                             inputdata& id) const override;
    const std::pair< std::vector<float>, std::vector<float> > get_weights_and_state(const std::vector<int>& query_trace, 
                                                                                            inputdata& id) const override;

  public:
    nn_weighted_output_sul(const std::string& cf) : nn_sul_base(cf) {};
    ~nn_weighted_output_sul();
};

#endif
