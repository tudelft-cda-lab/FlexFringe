/**
 * @file nn_discrete_and_float_output.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "nn_discrete_output_and_hidden_reps_sul.h"

#include <utility>

using namespace std;

#ifdef __FLEXFRINGE_PYTHON

/**
 * @brief IMPORTANT: **DO NOT DELETE**
 * 
 * The UNUSED function for retrieving the hidden reps of all symbols. For reference use if you or 
 * someone else in the future is interested in this.
 *
 * @param p_result The result as returned by the network. p_results assumed to be a flattened out matrix,
 * i.e. a vector that we have to infer the shape from again via HIDDEN_STATE_SIZE.
 * @param offset The initial offset where we find HIDDEN_STATE_SIZE.
 * @return vector< vector<double> > The hidden representations, one per input symbol. Usually including <SOS> and <EOS>.
 */
vector<vector<double>> compile_hidden_rep(PyObject* p_result, const int offset) {

    static const int HIDDEN_STATE_SIZE = static_cast<int>(PyLong_AsLong(
        PyList_GET_ITEM(p_result, static_cast<Py_ssize_t>(offset)))); // get first list, then return its length
    const int n_sequences = static_cast<int>((static_cast<int>(PyList_Size(p_result)) - 2) / HIDDEN_STATE_SIZE);
    vector<vector<double>> representations(n_sequences);
    for (int i = 0; i < n_sequences; ++i) {
        vector<double> hidden_rep(HIDDEN_STATE_SIZE);

        for (int j = 0; j < HIDDEN_STATE_SIZE; ++j) {
            int idx = i * HIDDEN_STATE_SIZE + j + offset + 1; // + offset + 1 because the first elements of p_result are
                                                              // predicted type, and eventually a confidence
            PyObject* s = PyList_GET_ITEM(p_result, static_cast<Py_ssize_t>(idx));
            hidden_rep[j] = static_cast<double>(PyFloat_AsDouble(s));
        }

        representations[i] = move(hidden_rep);
    }

    return representations;
}

/**
 * @brief Gets the hidden representation from the network. For description of p_result see do_query().
 */
vector<double> nn_discrete_output_and_hidden_reps_sul::compile_hidden_rep(PyObject* p_result) const {
    const static int res_size = PyList_Size(p_result);
    
    vector<double> representations(res_size - 1); // by convention: index 0 = prediction
    for (int idx = 1; idx < res_size; ++idx) { // + 1 cause we start at index 1
        PyObject* s = PyList_GET_ITEM(p_result, static_cast<Py_ssize_t>(idx));
        representations[idx] = static_cast<double>(PyFloat_AsDouble(s));
    }

    return representations;
}

/**
 * @brief by convention, python script must return a list. list[0]=prediction, and the rest of the returned 
 * list is the hidden representation of the last symbol of the query trace. The prediction can either 
 * be an integer or a string, whereas the string is obviously preferred.
 * 
 * It returns the prediction plus the hidden representation of the last symbol.
 * 
 */
const sul_response nn_discrete_output_and_hidden_reps_sul::do_query(const vector<int>& query_trace, inputdata& id) const {
    PyObject* p_list = PyList_New(query_trace.size());
    input_sequence_to_pylist(p_list, query_trace);

    PyObject* p_result = PyObject_CallOneArg(query_func, p_list);
    if (p_result==NULL || !PyList_Check(p_result))
        print_p_error();

    PyObject* p_type = PyList_GET_ITEM(p_result, static_cast<Py_ssize_t>(0));
    int type = pyobj_to_int(p_type, id);

    vector<double> hidden_rep = compile_hidden_rep(p_result);
    
    Py_DECREF(p_list);
    Py_DECREF(p_result);
    
    return sul_response(type, move(hidden_rep));
}

#else 

const sul_response nn_discrete_output_and_hidden_reps_sul::do_query(const vector<int>& query_trace, inputdata& id) const {
  nn_sul_base::do_query(query_trace, id);
}

#endif // __FLEXFRINGE_PYTHON