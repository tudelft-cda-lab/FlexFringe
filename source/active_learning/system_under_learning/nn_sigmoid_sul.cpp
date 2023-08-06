/**
 * @file nn_sigmoid_sul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-07-24
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "nn_sigmoid_sul.h"

using namespace std;

bool nn_sigmoid_sul::is_member(const std::list<int>& query_trace) const {
  return true;
}



const int nn_sigmoid_sul::query_trace(const std::list<int>& query_trace, inputdata& id) const {
 double nn_output = get_sigmoid_output(query_trace);
 int res = nn_output < 0.5 ? 0 : 1;
 return res;// TODO: make sure that there is no mismatch between the types
}

/**
 * @brief Does what you think it does.
 * 
 * @param query_trace 
 * @return const double 
 */
const double nn_sigmoid_sul::get_sigmoid_output(const std::list<int>& query_trace) const {
  PyObject* pylist = PyList_New(query_trace.size());
  for(int i=0; i < query_trace.size(); ++i){
    PyObject* pysymbol = PyLong_FromLong(query_trace.at(i));
    set_list_item(pylist, pysymbol, i);
  }

  PyObject* pyquery_result = PyObject_CallOneArg(query_func, pylist);
  return PyFloat_AsDouble(pyquery_result);
}

/**
 * @brief Like our conventional query_trace, but instead it returns the probability
 * of a string as assigned by the Network. Useful for e.g. the TAYSIR competition
 * (International Conference of Grammatical Inference, Rabat 2023)
 * 
 * @param query_trace 
 * @param id 
 * @return const float 
 */
const float nn_sigmoid_sul::get_sigmoid_output(const std::list<int>& query_trace, inputdata& id) const {
  assert(query_func != NULL);
  
}

/**
 * @brief Destroy the nn sigmoid sul::nn sigmoid sul object
 * 
 */
nn_sigmoid_sul::~nn_sigmoid_sul(){
  Py_Finalize();
}
