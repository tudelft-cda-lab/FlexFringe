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

class nn_weighted_output_sul : public nn_sul_base {
    friend class base_teacher;
    friend class eq_oracle_base;

  private:
    __attribute__((always_inline)) inline const double get_sigmoid_output(const std::vector<int>& query_trace,
                                                                          inputdata& id) const;

  protected:
    virtual void reset(){};
    virtual void init_types() const override;

    /* Learning from NN acceptors*/
    virtual bool is_member(const std::vector<int>& query_trace) const;
    virtual const int query_trace(const std::vector<int>& query_trace, inputdata& id) const override;
    virtual const std::pair< int, std::vector<float> > get_type_and_state(const std::vector<int>& query_trace, inputdata& id) const override;

    /* Learning from Language Models */
    virtual const double get_string_probability(const std::vector<int>& query_trace, inputdata& id) const override;
    virtual const std::vector<float> get_weight_distribution(const std::vector<int>& query_trace,
                                                             inputdata& id) const override;
    virtual const std::pair< std::vector<float>, std::vector<float> > get_weights_and_state(const std::vector<int>& query_trace, 
                                                                                            inputdata& id) const override;

  public:
    nn_weighted_output_sul() : nn_sul_base(){};
    ~nn_weighted_output_sul();
};

#endif
