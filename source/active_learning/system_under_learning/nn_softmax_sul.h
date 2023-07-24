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

#ifndef _NN_SOFTMAX_SUL_H_
#define _NN_SOFTMAX_SUL_H_

#include "sul_base.h"

class nn_softmax_sul {
  friend class base_teacher;
  friend class eq_oracle_base;

  private:
    cppflow::model model; // TODO: change the path, load in pre()

  protected:
    virtual void post() = 0;
    virtual void step() = 0;
    virtual void reset() = 0;

    virtual bool is_member(const std::list<int>& query_trace) const;
    virtual const int query_trace(const std::list<int>& query_trace, inputdata& id) const;
    
  public:
    nn_softmax_sul() = default; // abstract anyway

    void pre(inputdata& id);
};

#endif
