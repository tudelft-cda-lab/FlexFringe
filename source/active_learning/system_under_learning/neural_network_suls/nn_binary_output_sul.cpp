/**
 * @file nn_binary_output_sul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-12
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "nn_binary_output_sul.h"

#ifdef __FLEXFRINGE_PYTHON

using namespace std;

/**
 * @brief Does what you think it does.
 *
 * @param query_trace
 * @return const double
 */
const double nn_binary_output_sul::get_sigmoid_output(const vector<int>& query_trace, inputdata& id) const {
    PyObject* p_list = PyList_New(query_trace.size());
    input_sequence_to_pylist(p_list, query_trace);

    PyObject* p_query_result = PyObject_CallOneArg(query_func, p_list);
    if (p_query_result == NULL)
        print_p_error();

    double res = PyFloat_AsDouble(p_query_result);
    
    Py_DECREF(p_list);
    Py_DECREF(p_query_result);
    
    return res;
}


/**
 * @brief Returns a single binary output.
 *
 * @param query_trace
 * @param id
 * @return const int
 */
const sul_response nn_binary_output_sul::do_query(const vector<int>& query_trace, inputdata& id) const {
    const double nn_output = get_sigmoid_output(query_trace, id);
    if (nn_output < 0) {
        cerr << "Query trace: ";
        for (auto s : query_trace)
            cerr << s << " ";
        cerr << "\nOutput: " << nn_output << endl;

        print_p_error();

        throw runtime_error("Error in Python script, please check your code there. Potential Python error messages printed above.");
    }
    return nn_output < 0.5 ? sul_response(0) : sul_response(1);
}

#else

const sul_response nn_binary_output_sul::do_query(const std::vector<int>& query_trace, inputdata& id) const {
    nn_sul_base::do_query(query_trace, id);
}

#endif // __FLEXFRINGE_PYTHON
