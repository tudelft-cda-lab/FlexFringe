/**
 * @file sul_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Base class for the system under learning.
 * @version 0.1
 * @date 2023-02-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _SUL_BASE_H_
#define _SUL_BASE_H_

#include "misc/sqldb.h"
#include "source/input/inputdata.h"

#include <utility>
#include <stdexcept>
#include <fstream>

#include <vector>
#include <tuple>
#include <string>

class sul_base {
    friend class base_teacher;
    friend class eq_oracle_base;

  protected:
    virtual void reset() = 0;

    std::ifstream get_input_stream() const;

    virtual bool is_member(const std::vector<int>& query_trace) const = 0;
    virtual const int query_trace(const std::vector<int>& query_trace, inputdata& id) const = 0;
    virtual const std::pair< int, std::vector< std::vector<float> > > get_type_and_states(const std::vector<int>& query_trace, inputdata& id) const;
    virtual const std::tuple< std::string, float, std::vector< std::vector<float> > > get_type_confidence_and_states(const std::vector<int>& query_trace, inputdata& id) const;
    virtual const std::vector< std::pair<std::string, float> > get_type_confidence_batch(const std::vector< std::vector<int> >& query_traces, inputdata& id) const;

    // TODO: can we simplify all of those with some sort of template?
    virtual const double get_string_probability(const std::vector<int>& query_trace, inputdata& id) const;
    virtual const std::vector<float> get_weight_distribution(const std::vector<int>& query_trace, inputdata& id) const;
    virtual const std::pair< std::vector<float>, std::vector<float> > get_weights_and_state(const std::vector<int>& query_trace, inputdata& id) const;
  
  public:
    /**
     * @brief Initialize the sul class.
     */
    virtual void pre(inputdata& id) = 0;
};

#endif
