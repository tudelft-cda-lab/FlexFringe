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
#include <cassert>
#include <unordered_map>

#ifdef __FLEXFRINGE_PYTHON

#define PY_SSIZE_T_CLEAN // recommended, see https://docs.python.org/3/extending/extending.html#a-simple-example
#include <Python.h> // IMPORTANT: Python.h must be first import. See https://docs.python.org/3/extending/extending.html

class nn_sul_base : public sul_base {
    friend class oracle_base;

  protected:
    const std::string CONNECTOR_FILE;

    PyObject* p_module;
    PyObject* p_model_path;
    PyObject* query_func;
    PyObject* load_model_func;

    const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) const = 0;
    void reset() = 0;

    void input_sequence_to_pylist(PyObject* p_list_out, const std::vector<int>& c_list) const;
    int pyobj_to_int(PyObject* p_obj, inputdata& id) const;
    void print_p_error() const;

    virtual void init_types() const; // we need to set the internal types of flexfringe to match the neural network

  public:
    void pre(inputdata& id) override;

    nn_sul_base() = default;
    ~nn_sul_base();
};

/* Dummy implementation when Python disabled to get it to compile on platforms without Python Dev Headers. */
#else

class nn_sul_base : public sul_base {
    friend class oracle_base;

  public:
    const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) const override;
    const sul_response do_query(const std::vector< std::vector<int> >& query_trace, inputdata& id) const override;

    void reset() override {};

    void pre(inputdata& id) override;
    
    nn_sul_base() {
      throw std::logic_error("Neural network SULs can only be used with Python flag enabled");
    };
};

#endif /* __FLEXFRINGE_PYTHON */
#endif
