/**
 * @file nn_sigmoid_sul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * https://serizba.github.io/cppflow/quickstart.html#load-a-model
 * @version 0.1
 * @date 2023-07-24
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "nn_sigmoid_sul.h"

#define PY_SSIZE_T_CLEAN // recommended, see https://docs.python.org/3/extending/extending.html#a-simple-example
#include <Python.h>

using namespace std;

bool nn_sigmoid_sul::is_member(const std::list<int>& query_trace) const {
  return true;
}

const int nn_sigmoid_sul::query_trace(const std::list<int>& query_trace, inputdata& id) const {
  
  auto output = model(input);

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

}

/**
 * @brief Destroy the nn sigmoid sul::nn sigmoid sul object
 * 
 */
nn_sigmoid_sul::~nn_sigmoid_sul(){
  Py_Finalize();
}
