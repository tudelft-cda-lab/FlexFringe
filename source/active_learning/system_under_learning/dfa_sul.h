/**
 * @file dfa_sul.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief We use this class to learn from DFAs we have previously written in flexfringe, or another DFA that supports the parser. 
 * Useful e.g. for verification of new algorithms.
 * @version 0.1
 * @date 2024-11-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _DFA_SUL_H_
#define _DFA_SUL_H_

#include "apta.h"
#include "sul_base.h"

#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

class dfa_sul : public sul_base {
  private:
    std::unique_ptr<apta> sut; // an apta representing the system under test

  public:
    dfa_sul() { sut = std::unique_ptr<apta>(new apta()); }
    void pre(inputdata& id) override;

    void reset() override{};
    const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) const override;
};

#endif
