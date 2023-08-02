/**
 * @file sul_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This SUL connect with an NN. The NN is called from within a Python script (that we also provide a template
 * for), and the values are returned to us here.
 * @version 0.1
 * @date 2023-02-15
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _NN_SIGMOID_SUL_H_
#define _NN_SIGMOID_SUL_H_

#include "nn_sul_base.h"

class nn_sigmoid_sul : nn_sul_base {
  friend class base_teacher;
  friend class eq_oracle_base;

  protected:
    virtual void post(){};
    virtual void step(){};
    virtual void reset(){};

    virtual bool is_member(const std::list<int>& query_trace) const;
    virtual const int query_trace(const std::list<int>& query_trace, inputdata& id) const;
    
  public:
    nn_sigmoid_sul() : PYTHON_MODULE_NAME("nn_connector_sigmoid.py"){}; // abstract anyway
    ~nn_sigmoid_sul();
};

#endif
