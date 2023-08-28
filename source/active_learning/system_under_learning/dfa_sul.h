/**
 * @file dfa_sul.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Base class for the system under learning.
 * @version 0.1
 * @date 2023-02-15
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _DFA_SUL_H_
#define _DFA_SUL_H_

#include "apta.h"
#include "sul_base.h"

#include <fstream>
#include <vector>
#include <stdexcept>
#include <memory>

class dfa_sul : public sul_base {
  friend class base_teacher;
  friend class eq_oracle_base;

  private:
    std::unique_ptr<apta> sut;

  protected:
    virtual void post() override {};
    virtual void step() override {};
    virtual void reset() override {};

    virtual bool is_member(const std::vector<int>& query_trace) const override;
    virtual const int query_trace(const std::vector<int>& query_trace, inputdata& id) const override;

  public:
    dfa_sul(){
      sut = std::unique_ptr<apta>( new apta() );
    }

    virtual void pre(inputdata& id) override;
    void pre(inputdata& id, const bool overwrite_apta);
};

#endif
