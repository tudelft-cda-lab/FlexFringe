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
#include "inputdatalocator.h"

#include <unordered_set>
#include <string>
#include <stdexcept>

using namespace std;

bool nn_sigmoid_sul::is_member(const std::list<int>& query_trace) const {
  return true;
}


/**
 * @brief Queries trace, returns result.
 * 
 * Side effect: Check if the inferred type is part of the types in inputdata. If no, set.
 * 
 * @param query_trace 
 * @param id 
 * @return const int 
 */
const int nn_sigmoid_sul::query_trace(const std::list<int>& query_trace, inputdata& id) const {
  double nn_output = get_sigmoid_output(query_trace, id);
  if(nn_output < 0){
    throw runtime_error("Error in Python script, please check your code there.");
  }
  
  int res = nn_output < 0.5 ? 0 : 1;
  return res;// TODO: make sure that there is no mismatch between the types
}

/**
 * @brief Initialize the types to 0 and 1. Which is which depends on how the network was trained.
 * 
 */
void nn_sigmoid_sul::init_types() const {
  inputdata_locator::get()->add_type(std::string("Type 0"));
  inputdata_locator::get()->add_type(std::string("Type 1"));
}

/**
 * @brief Does what you think it does.
 * 
 * @param query_trace 
 * @return const double 
 */
const double nn_sigmoid_sul::get_sigmoid_output(const std::list<int>& query_trace, inputdata& id) const {
  PyObject* p_list = PyList_New(query_trace.size());
  int i = 0;
  for(const int symbol: query_trace){
    std::string mapped_symbol = id.get_symbol(symbol);
    PyObject* p_symbol = PyUnicode_FromString(mapped_symbol.c_str());
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
