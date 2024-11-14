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

#include "nn_discrete_and_float_output.h"

#include <utility>

using namespace std;

#ifdef __FLEXFRINGE_PYTHON

/**
 * @brief Be careful what this function expects the network to return. A single list of size 2: [int, float]
 * 
 */
const sul_response nn_discrete_output_sul::do_query(const vector<int>& query_trace, inputdata& id) const {
    PyObject* p_list = PyList_New(query_trace.size());
    input_sequence_to_pylist(p_list, query_trace);

    PyObject* p_result = PyObject_CallOneArg(query_func, p_list);
    if (p_result==NULL || !PyList_Check(p_result))
        print_p_error();

    PyObject* p_type = PyList_GET_ITEM(p_result, static_cast<Py_ssize_t>(0));
    int type = pyobj_to_int(p_type, id);

    PyObject* p_confidence = PyList_GET_ITEM(p_result, static_cast<Py_ssize_t>(1));
    if (!PyFloat_CheckExact(p_confidence)) {
        cerr << "Problem with type as returned by Python script. Is it a proper float?" << endl;
        throw exception(); // force the catch block
    }

    float confidence = static_cast<float>(PyFloat_AsDouble(p_confidence));
    
    Py_DECREF(p_list);
    Py_DECREF(p_result);

    return sul_response(type, confidence);
}

/**
 * @brief Be careful what this function expects the network to return. A single list, with corresponding int/float pairs 
 * consecutively. Meaning, the following applies: [int1, float1, int2, float2, int3, float3, ...]
 * 
 * TODO: Possibly not efficient due to fragmentation in int/float. 
 * You can customize this with another SUL, where you e.g. return a dictionary or first have all ints, then all floats.
 */
const sul_response nn_discrete_output_sul::do_query(const vector< vector<int> >& query_traces, inputdata& id) const {
    PyObject* p_list = PyList_New(query_traces.size());
    for(int i=0; i<query_traces.size(); i++){
        PyObject* p_tmp = PyList_New(query_traces[i].size());
        input_sequence_to_pylist(p_tmp, query_traces[i]);
        PyList_SET_ITEM(p_list, i, p_tmp);
    }

    PyObject* p_result = PyObject_CallOneArg(query_func, p_list);
    if (p_result == NULL || !PyList_Check(p_result))
        print_p_error();

    vector<int> int_res(query_traces.size());
    vector<float> float_res(query_traces.size());
    for(int i=0; i<query_traces.size(); i++){

        PyObject* p_type = PyList_GET_ITEM(p_result, static_cast<Py_ssize_t>(i*2));
        int type = pyobj_to_int(p_type, id);

        PyObject* p_confidence = PyList_GET_ITEM(p_result, static_cast<Py_ssize_t>(i*2 + 1));
        if(!PyFloat_CheckExact(p_confidence)){
            cerr << "Problem with type as returned by Python script. Is it a proper float?" << endl;
            throw exception(); // force the catch block
        }

        int_res[i] = type;
        float_res[i] = static_cast<float>(PyFloat_AsDouble(p_confidence));
    }

    Py_DECREF(p_list);
    Py_DECREF(p_result);

    return sul_response(move(int_res), move(float_res));
}

#else

const sul_response nn_discrete_output_sul::do_query(const vector<int>& query_trace, inputdata& id) const {
  nn_sul_base::do_query(query_trace, id);
}

const sul_response nn_discrete_output_sul::do_query(const vector< vector<int> >& query_traces, inputdata& id) const {
  nn_sul_base::do_query(query_traces, id);
}

#endif // __FLEXFRINGE_PYTHON