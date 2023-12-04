/**
 * @file nn_weighted_output_sul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-07-24
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "nn_weighted_output_sul.h"
#include "inputdatalocator.h"

#include <unordered_set>
#include <string>
#include <stdexcept>

using namespace std;

bool nn_weighted_output_sul::is_member(const std::vector<int>& query_trace) const {
  return true;
}


/**
 * @brief Queries trace, returns result.
 * 
 * Sidenote: We have duplicate code with query_trace here. The reason being is that is is a hot
 * path, hence we avoid function call overhead with this as part of our optimization. -> duplicate code 
 * is wanted. 
 * 
 * @param query_trace 
 * @param id 
 * @return const double The probability that this string belongs to the language learned by the model. 
 */
const double nn_weighted_output_sul::get_string_probability(const std::vector<int>& query_trace, inputdata& id) const {
  const double nn_output = get_sigmoid_output(query_trace, id);
  if(nn_output < 0){
    throw runtime_error("Error in Python script, please check your code there.");
  }
  
  return nn_output;
}

/**
 * @brief Queries trace, returns result.
 * 
 * Sidenote: We have duplicate code with get_string_probability here. The reason being is that is is a hot
 * path, hence we avoid function call overhead with this as part of our optimization. -> duplicate code 
 * is wanted. 
 * 
 * @param query_trace 
 * @param id 
 * @return const int 
 */
const int nn_weighted_output_sul::query_trace(const std::vector<int>& query_trace, inputdata& id) const {
  const double nn_output = get_sigmoid_output(query_trace, id);
  if(nn_output < 0){
    throw runtime_error("Error in Python script, please check your code there.");
  }
  return nn_output < 0.5 ? 0 : 1;
}

/**
 * @brief Initialize the types to 0 and 1. Which is which depends on how the network was trained.
 * 
 */
void nn_weighted_output_sul::init_types() const {
  inputdata_locator::get()->add_type(std::string("Type 0"));
  inputdata_locator::get()->add_type(std::string("Type 1"));
}

/**
 * @brief Does what you think it does.
 * 
 * @param query_trace 
 * @return const double 
 */
const double nn_weighted_output_sul::get_sigmoid_output(const std::vector<int>& query_trace, inputdata& id) const {

  static PyObject* p_start_symbol = START_SYMBOL == -1 ? nullptr : PyLong_FromLong(START_SYMBOL); 
  static PyObject* p_end_symbol = START_SYMBOL == -1 ? nullptr : PyLong_FromLong(END_SYMBOL); ;

  PyObject* p_list = p_start_symbol==nullptr ? PyList_New(query_trace.size()) : PyList_New(query_trace.size()+2); // +2 for start and end symbol
  int i = p_start_symbol==nullptr ? 0 : 1;
  for(const int flexfringe_symbol: query_trace){
    int mapped_symbol = input_mapper.at(flexfringe_symbol);
    PyObject* p_symbol = PyLong_FromLong(mapped_symbol);
    set_list_item(p_list, p_symbol, i);
    ++i;
  }

  if(p_start_symbol!=nullptr){
    set_list_item(p_list, p_start_symbol, 0);
    set_list_item(p_list, p_end_symbol, query_trace.size()+1);
  }

  PyObject* p_query_result = PyObject_CallOneArg(query_func, p_list);
  return PyFloat_AsDouble(p_query_result);
}

/**
 * @brief Get the weight distribution, i.e. all the next symbol probabilities/weights for a given sequence.
 * 
 * @param query_trace The query trace from which we want to pick up the weights.
 * @param id Inputdata
 * @return const std::vector<double> A vector with the weights. Vector size is that of alphabet + 2 if trained with 
 * <SOS> and <EOS> token. These have to be dealt with by the algorithm.
 */
const std::vector<float> nn_weighted_output_sul::get_weight_distribution(const std::vector<int>& query_trace, inputdata& id) const {
  static PyObject* p_start_symbol = START_SYMBOL == -1 ? nullptr : PyLong_FromLong(START_SYMBOL); 
  static PyObject* p_end_symbol = END_SYMBOL == -1 ? nullptr : PyLong_FromLong(END_SYMBOL); ;

  PyObject* p_list = p_start_symbol==nullptr ? PyList_New(query_trace.size()) : PyList_New(query_trace.size()+1);
  int i = p_start_symbol==nullptr ? 0 : 1;
  for(const int flexfringe_symbol: query_trace){
    int mapped_symbol = input_mapper.at(flexfringe_symbol);
    PyObject* p_symbol = PyLong_FromLong(mapped_symbol);
    set_list_item(p_list, p_symbol, i);
    ++i;
    //Py_DECREF(p_symbol);
  }

  if(p_start_symbol!=nullptr){
    set_list_item(p_list, p_start_symbol, 0);
  }

  PyObject* p_weights = PyObject_CallOneArg(query_func, p_list);
  Py_DECREF(p_list);
  if(!PyList_CheckExact(p_weights))
    throw std::runtime_error("Something went wrong, the Network did not return a list. What happened?");
  
  static const int RESPONSE_SIZE = static_cast<int>(PyList_Size(p_weights));
  vector<float> res(RESPONSE_SIZE);
  for(int i=0; i<RESPONSE_SIZE; ++i){
    PyObject* resp = PyList_GetItem(p_weights, static_cast<Py_ssize_t>(i));
    res[i] = static_cast<float>(PyFloat_AsDouble(resp));
    Py_DECREF(resp);
  }
  Py_DECREF(p_weights);
  return res;
}

/**
 * @brief Destroy the nn sigmoid sul::nn sigmoid sul object
 * 
 */
nn_weighted_output_sul::~nn_weighted_output_sul(){
  Py_Finalize();
}
