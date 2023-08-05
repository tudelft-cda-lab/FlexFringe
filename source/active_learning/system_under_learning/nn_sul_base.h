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

#include <string>

#define PY_SSIZE_T_CLEAN // recommended, see https://docs.python.org/3/extending/extending.html#a-simple-example
#include <Python.h>

class nn_sul_base : sul_base {
  friend class base_teacher;
  friend class eq_oracle_base;

  protected:
    const std::string PYTHON_SCRIPT_PATH = "python"; // relative path to where python scripts are
    const std::string PYTHON_MODULE_NAME = "";

    PyObject* pModule;
    PyObject* query_func;
    PyObject* alphabet_func;

    virtual void post() = 0;
    virtual void step() = 0;
    virtual void reset() = 0;

    virtual bool is_member(const std::list<int>& query_trace) const = 0;
    virtual const int query_trace(const std::list<int>& query_trace, inputdata& id) const = 0;
    
  public:
    nn_sul_base() = default; // abstract anyway

    virtual void pre(inputdata& id) override;
};

#endif
