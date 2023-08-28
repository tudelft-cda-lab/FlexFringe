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

#include <string>
#include <unordered_map>

#define PY_SSIZE_T_CLEAN // recommended, see https://docs.python.org/3/extending/extending.html#a-simple-example
#include <Python.h>

class nn_sul_base : public sul_base {
  friend class base_teacher;
  friend class eq_oracle_base;

  protected:
    nn_sul_base() = default;

    PyObject* p_module;
    PyObject* query_func;
    PyObject* alphabet_func;
    PyObject* load_model_func;

    std::unordered_map<int, int> input_mapper;

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
