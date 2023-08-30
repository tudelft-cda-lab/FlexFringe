/**
 * @file nn_sul_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Base for neural network queries.
 * 
 * Documentation of CPython-API: https://docs.python.org/3/c-api/index.html

 * @version 0.1
 * @date 2023-08-02
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _NN_SUL_BASE_H_
#define _NN_SUL_BASE_H_

#include "sul_base.h"
#include "parameters.h"

#include <string>
#include <unordered_map>
#include <cassert>

#define PY_SSIZE_T_CLEAN // recommended, see https://docs.python.org/3/extending/extending.html#a-simple-example
#include <Python.h>

class nn_sul_base : public sul_base {
  friend class base_teacher;
  friend class eq_oracle_base;

  protected:
    nn_sul_base() : start_symbol(START_SYMBOL), end_symbol(END_SYMBOL) {
      assert(( (void*) "If <SOS> or <EOS> is set (!=-1), then the other one must be set, too.", (start_symbol==-1 && end_symbol==-1) || (start_symbol!=-1 && end_symbol!=-1) ));
      if(start_symbol!=-1){
        p_start_symbol = PyLong_FromLong(start_symbol);
        p_end_symbol = PyLong_FromLong(end_symbol);
      }
    };

    PyObject* p_module;
    PyObject* query_func;
    PyObject* alphabet_func;
    PyObject* load_model_func;

    std::unordered_map<int, int> input_mapper;
    
    const int start_symbol;
    const int end_symbol;
    PyObject* p_start_symbol; 
    PyObject* p_end_symbol;

    virtual void post() = 0;
    virtual void step() = 0;
    virtual void reset() = 0;

    virtual bool is_member(const std::vector<int>& query_trace) const = 0;
    virtual const int query_trace(const std::vector<int>& query_trace, inputdata& id) const = 0;
    
    void set_list_item(PyObject* pylist, PyObject* item, const int idx) const;
    virtual void init_types() const = 0; // we need to set the internal types of flexfringe according to the types we expect

  public:
    virtual void pre(inputdata& id) override;
};

#endif
