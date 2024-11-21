/**
 * @file sul_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Implementation of SUL base class.
 * @version 0.1
 * @date 2023-02-19
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "sul_base.h"
#include "parameters.h"

#include <iostream>
#include <stdexcept>

using namespace std;

/**
  * @brief Get the bool response. Safety check included.
  */
bool sul_response::get_bool() const {
    if(!bool_opt){
      throw runtime_error("Tried to retrieve bool response, but it does not exist in response type.");
    }
    return bool_opt.value();
}

/**
 * @brief Same as get_int, but no check on whether optional has been set. Will terminate program if called unsuccessfully.
 */
bool sul_response::GET_BOOL() const noexcept {
    return bool_opt.value();
}

/**
  * @brief Get the int response. Safety check included.
  */
int sul_response::get_int() const {
    if(!int_opt){
      throw runtime_error("Tried to retrieve integer response, but it does not exist in response type.");
    }
    return int_opt.value();
}

/**
 * @brief Same as get_int, but no check on whether optional has been set. Will terminate program if called unsuccessfully.
 */
int sul_response::GET_INT() const noexcept {
    return int_opt.value();
}

/**
 * @brief Get the int response. Safety check included.
 */
double sul_response::get_double() const {
    if(!double_opt){
      throw runtime_error("Tried to retrieve double response, but it does not exist in response type.");
    }
    return double_opt.value();
}

/**
 * @brief Same as get_double, but no check on whether optional has been set. Will terminate program if called unsuccessfully.
 */
double sul_response::GET_DOUBLE() const noexcept {
  return double_opt.value();
}

/**
 * @brief Get the vector with ints. Safety check included.
 */
const vector<int>& sul_response::get_int_vec() const {
  if(!int_vec_opt){
    throw runtime_error("Tried to retrieve int-vec field, but it does not exist in response type.");
  }
  return int_vec_opt.value();
}

/**
 * @brief Same as get_int_vec, but no check on whether optional has been set. Will terminate program if called unsuccessfully.
 */
const vector<int>& sul_response::GET_INT_VEC() const noexcept {
  return int_vec_opt.value();
}

/**
 * @brief Get the vector with floats. Safety check included.
 */
const vector<double>& sul_response::get_double_vec() const {
  if(!double_vec_opt){
    throw runtime_error("Tried to retrieve double-vec field, but it does not exist in response type.");
  }
  return double_vec_opt.value();
}

/**
 * @brief Same as get_double_vec, but no check on whether optional has been set. Will terminate program if called unsuccessfully.
 */
const vector<double>& sul_response::GET_DOUBLE_VEC() const noexcept {
  return double_vec_opt.value();
}

/**
 * @brief Returns the stream to an input file, provided by flag INPUT_FILE.
 */
ifstream sul_base::get_input_stream() const {
    ifstream input_stream(INPUT_FILE);
    cout << "Input file: " << INPUT_FILE << endl;

    if (!input_stream) {
        cerr << "Input file not found, aborting" << endl;
        exit(-1);
    } else {
        cout << "Using input file: " << INPUT_FILE << endl;
    }
    return input_stream;
}