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
float sul_response::get_float() const {
    if(!float_opt){
      throw runtime_error("Tried to retrieve float response, but it does not exist in response type.");
    }
    return float_opt.value();
}

/**
 * @brief Same as get_float, but no check on whether optional has been set. Will terminate program if called unsuccessfully.
 */
float sul_response::GET_FLOAT() const noexcept {
  return float_opt.value();
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
const vector<float>& sul_response::get_float_vec() const {
  if(!float_vec_opt){
    throw runtime_error("Tried to retrieve float-vec field, but it does not exist in response type.");
  }
  return float_vec_opt.value();
}

/**
 * @brief Same as get_float_vec, but no check on whether optional has been set. Will terminate program if called unsuccessfully.
 */
const vector<float>& sul_response::GET_FLOAT_VEC() const noexcept {
  return float_vec_opt.value();
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

const int sul_base::query_trace(const vector<int>& query_trace, inputdata& id) const {
    throw logic_error("query_trace not implemented. That is a programming error. Aborting program.");
}

const pair<int, vector< vector<float> > > sul_base::get_type_and_states(const vector<int>& query_trace, inputdata& id) const {
    throw logic_error("This SUL does not support type queries along with a hidden state representation. \
    Please change the program settings. Aborting program.");
}

const tuple<int, float, vector< vector<float> > > sul_base::get_type_confidence_and_states(const vector<int>& query_trace, inputdata& id) const {
    throw logic_error("This SUL does not support type queries along with confidence and hidden state representation. \
    Please change the program settings. Aborting program.");
}

const vector< pair<int, float> > sul_base::get_type_confidence_batch(const vector< vector<int> >& query_traces, inputdata& id) const {
    throw logic_error("This SUL does not support type queries along with confidence and hidden state representation. \
    Please change the program settings. Aborting program.");
}

const double sul_base::get_string_probability(const vector<int>& query_trace, inputdata& id) const {
    throw logic_error("The SUL tries to infer string probability, but this SUL does not support this. \
  Please change the program settings. Aborting program.");
}

const vector<float> sul_base::get_weight_distribution(const vector<int>& query_trace, inputdata& id) const {
    throw logic_error(
        "This SUL does not support inference of the weight distribution. Please change the program settings. \
  Aborting program.");
};

const pair< vector<float>, vector<float> > sul_base::get_weights_and_state(const vector<int>& query_trace, inputdata& id) const{
    throw logic_error(
        "This SUL does not support inference of the weight distribution. Please change the program settings. \
  Aborting program.");  
}

