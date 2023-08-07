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
  PyObject* p_list = PyList_New(query_trace.size());
  int i = 0;
  for(const int symbol: query_trace){
    PyObject* p_symbol = PyLong_FromLong(symbol);
    set_list_item(p_list, p_symbol, i);
    ++i;
  }

  PyObject* p_query_result = PyObject_CallOneArg(query_func, p_list);
  return PyFloat_AsDouble(p_query_result);
}

/**
 * @brief Destroy the nn sigmoid sul::nn sigmoid sul object
 * 
 */
nn_sigmoid_sul::~nn_sigmoid_sul(){
  Py_Finalize();
}
