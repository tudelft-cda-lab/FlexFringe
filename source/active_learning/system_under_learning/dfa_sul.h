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

#include "inputdata.h"
#include "apta.h"

#include <fstream>
#include <vector>
#include <stdexcept>
#include <memory>

class dfa_sul : public sul_base {
  friend class base_teacher;
  friend class eq_oracle_base;

  private:
    std::unique_ptr<apta> sul;

  protected:
    virtual void post(){};
    virtual void step(){};

    virtual void reset(){};

    virtual bool is_member(const std::vector<int>& query_trace) const override;
    virtual const int query_trace(const std::vector<int>& query_trace, inputdata& id) const override;

  public:
    dfa_sul(){
      sul = std::unique_ptr<apta>( new apta() );
    };

    virtual void pre(inputdata& id) override;
};

#endif
