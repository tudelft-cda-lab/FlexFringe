/**
 * @file nn_float_output_sul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "nn_float_output_sul.h"

using namespace std;

#ifdef __FLEXFRINGE_PYTHON

/**
 * @brief Getting a single float from the network, or a list of size 1 containing a float.
 */
const sul_response nn_float_output_sul::do_query(const vector<int>& query_trace, inputdata& id) const {
    PyObject* p_list = PyList_New(query_trace.size());
    input_sequence_to_pylist(p_list, query_trace);

    PyObject* p_result = PyObject_CallOneArg(query_func, p_list);
    if (p_result == NULL)
        print_p_error();

    float res;
    if(PyList_Check(p_result) && PyList_Size(p_result) == 1){
      PyObject* p_float = PyList_GET_ITEM(p_result, 0);
      if(!PyFloat_Check(p_float)){
        cerr << "Error in return value of Python script. The list did not contain a proper float value. Terminating." << endl;
        exit(EXIT_FAILURE);
      }
      res = static_cast<float>(PyFloat_AsDouble(p_float));
    }
    else if(PyFloat_Check(p_result)){
      res = static_cast<float>(PyFloat_AsDouble(p_result));
    }
    else{
      throw std::runtime_error("Python script neither returned a list of length 1 nor a float value"); 
    }
    
    Py_DECREF(p_list);
    Py_DECREF(p_result);
    
    return sul_response(res);
}

/**
 * @brief Expects the network to return the integer values in a list.
 */
const sul_response nn_float_output_sul::do_query(const vector< vector<int> >& query_traces, inputdata& id) const {
    PyObject* p_list = PyList_New(query_traces.size());
    for(int i=0; i<query_traces.size(); i++){
        PyObject* p_tmp = PyList_New(query_traces[i].size());
        input_sequence_to_pylist(p_tmp, query_traces[i]);
        PyList_SET_ITEM(p_list, i, p_tmp);
    }

    PyObject* p_result = PyObject_CallOneArg(query_func, p_list);
    if (p_result == NULL || !PyList_Check(p_result))
        print_p_error();

    vector<float> res(query_traces.size());
    for(int i=0; i<query_traces.size(); i++){

        PyObject* p_float = PyList_GET_ITEM(p_result, static_cast<Py_ssize_t>(i));
        if(!PyFloat_Check(p_type)){
            cerr << "Problem with type as returned by Python script. Is it a proper float?" << endl;
            throw exception(); // force the catch block
        }

        res[i] = PyFloat_AsDouble(p_float);
    }

    Py_DECREF(p_list);
    Py_DECREF(p_result);

  return sul_response(move(res));
}

#else

const sul_response nn_float_output_sul::do_query(const vector<int>& query_trace, inputdata& id) const {
  nn_sul_base::do_query(query_trace, id);
}

const sul_response nn_float_output_sul::do_query(const vector< vector<int> >& query_traces, inputdata& id) const {
  nn_sul_base::do_query(query_traces, id);
}

#endif // __FLEXFRINGE_PYTHON