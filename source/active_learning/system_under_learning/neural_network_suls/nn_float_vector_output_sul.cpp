/**
 * @file nn_float_vector_output_sul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "nn_float_vector_output_sul.h"

using namespace std;

#ifdef __FLEXFRINGE_PYTHON

/**
 * @brief Getting a vector of floats from the network, and returning that.
 */
const sul_response nn_float_vector_output_sul::do_query(const vector<int>& query_trace, inputdata& id) const {
    PyObject* p_list = PyList_New(query_trace.size());
    input_sequence_to_pylist(p_list, query_trace);

    PyObject* p_result = PyObject_CallOneArg(query_func, p_list);
    if (p_result == NULL)
        print_p_error();
    
    if(!PyList_Check(p_result))
      throw std::runtime_error("Python script did not return a list to SUL that expected a list.");

    const static int res_size = PyList_Size(p_result);
    vector<float> res(res_size);
    for(int i=0; i<res_size; ++i){
      PyObject* p_float = PyList_GET_ITEM(p_result, i);
      if(!PyFloat_Check(p_float)){
        cerr << "Error in return value of Python script. The list did not contain a proper float value. Terminating." << endl;
        exit(EXIT_FAILURE);
      }

      res[i] = static_cast<float>(PyFloat_AsDouble(p_float));
    }
    
    Py_DECREF(p_list);
    Py_DECREF(p_result);
    
    return sul_response(move(res));
}

#else

const sul_response nn_float_vector_output_sul::do_query(const vector<int>& query_trace, inputdata& id) const {
  nn_sul_base::do_query(query_trace, id);
}

#endif // __FLEXFRINGE_PYTHON