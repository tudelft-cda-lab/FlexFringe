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

#include <stdexcept>
#include <string>
#include <unordered_set>

#include <iostream>

using namespace std;

bool nn_weighted_output_sul::is_member(const std::vector<int>& query_trace) const { return true; }

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
    if (nn_output < 0) {
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
    if (nn_output < 0) {
        cerr << "Query trace: ";
        for(auto s: query_trace) cerr << s << " ";
        cerr << "\nOutput: " << nn_output << endl;
        throw runtime_error("Error in Python script, please check your code there.");
    }
    return nn_output < 0.5 ? 0 : 1;
}

/**
 * @brief Gets the type, assumed to be of Sigmoid output and a threshold of 0.5, along with 
 * a hidden representation of the network state.
 * 
 * @return const std::pair< int, std::vector<float> > <the type, the hidden representation>
 */
const std::pair< int, std::vector<float> > 
nn_weighted_output_sul::get_type_and_state(const std::vector<int>& query_trace, inputdata& id) const {
    static PyObject* p_start_symbol = START_SYMBOL == -1 ? nullptr : PyLong_FromLong(START_SYMBOL);
    static PyObject* p_end_symbol = START_SYMBOL == -1 ? nullptr : PyLong_FromLong(END_SYMBOL);

    PyObject* p_list = p_start_symbol == nullptr ? PyList_New(query_trace.size())
                                                 : PyList_New(query_trace.size() + 2); // +2 for start and end symbol
    int i = p_start_symbol == nullptr ? 0 : 1;
    for (const int flexfringe_symbol : query_trace) {
        int mapped_symbol = stoi(id.get_symbol(flexfringe_symbol));
        PyObject* p_symbol = PyLong_FromLong(mapped_symbol);
        set_list_item(p_list, p_symbol, i);
        ++i;
    }

    if (p_start_symbol != nullptr) {
        Py_INCREF(p_start_symbol); // needed because PyList_SetItem hands ownership to p_list, see https://docs.python.org/3/extending/extending.html#ownership-rules
        Py_INCREF(p_end_symbol);
        set_list_item(p_list, p_start_symbol, 0);
        set_list_item(p_list, p_end_symbol, query_trace.size() + 1);
    }

    PyObject* p_query_result = PyObject_CallOneArg(query_func, p_list);
    PyFloat_AsDouble(p_query_result);

    PyObject* p_result = PyObject_CallOneArg(query_func, p_list);
    if (!PyTuple_Check(p_result))
        throw std::runtime_error("Something went wrong, the Network did not return a tuple. What happened?");
    assert(static_cast<int>(PyTuple_Size(p_result)) == 2);

    PyObject* p_type = PyTuple_GET_ITEM(p_result, static_cast<Py_ssize_t>(0));
    PyObject* p_state = PyTuple_GET_ITEM(p_result, static_cast<Py_ssize_t>(1));

    static const int STATE_SIZE = static_cast<int>(PyList_Size(p_state));    
    vector<float> state(STATE_SIZE);
    for (int i = 0; i < STATE_SIZE; ++i) {
        PyObject* s = PyList_GET_ITEM(p_state, static_cast<Py_ssize_t>(i));
        state[i] = static_cast<float>(PyFloat_AsDouble(s));
    }

    if(!PyFloat_Check(p_type)){
        throw std::runtime_error("Something went wrong, the Network neither returned a float (binary acceptor model\
        , nor did it return a list (language model)). What happened?");
    }
    
    int type = static_cast<float>(PyFloat_AsDouble(p_type)) < 0.5 ? 0 : 1;
    return make_pair(type, state);
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
    static PyObject* p_end_symbol = START_SYMBOL == -1 ? nullptr : PyLong_FromLong(END_SYMBOL);

    PyObject* p_list = p_start_symbol == nullptr ? PyList_New(query_trace.size())
                                                 : PyList_New(query_trace.size() + 2); // +2 for start and end symbol
    int i = p_start_symbol == nullptr ? 0 : 1;
    for (const int flexfringe_symbol : query_trace) {
        int mapped_symbol = stoi(id.get_symbol(flexfringe_symbol));
        PyObject* p_symbol = PyLong_FromLong(mapped_symbol);
        set_list_item(p_list, p_symbol, i);
        ++i;
    }

    if (p_start_symbol != nullptr) {
        Py_INCREF(p_start_symbol); // needed because PyList_SetItem hands ownership to p_list, see https://docs.python.org/3/extending/extending.html#ownership-rules
        Py_INCREF(p_end_symbol);
        set_list_item(p_list, p_start_symbol, 0);
        set_list_item(p_list, p_end_symbol, query_trace.size() + 1);
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
const std::vector<float> nn_weighted_output_sul::get_weight_distribution(const std::vector<int>& query_trace,
                                                                         inputdata& id) const {
    static PyObject* p_start_symbol = START_SYMBOL == -1 ? nullptr : PyLong_FromLong(START_SYMBOL);
    static PyObject* p_end_symbol = END_SYMBOL == -1 ? nullptr : PyLong_FromLong(END_SYMBOL);

    PyObject* p_list = p_start_symbol == nullptr ? PyList_New(query_trace.size()) : PyList_New(query_trace.size() + 1);
    int i = p_start_symbol == nullptr ? 0 : 1;
    for (const int flexfringe_symbol : query_trace) {
        int mapped_symbol = stoi(id.get_symbol(flexfringe_symbol));
        PyObject* p_symbol = PyLong_FromLong(mapped_symbol);
        PyList_SetItem(p_list, static_cast<Py_ssize_t>(i), p_symbol);
        ++i;
    }

    if (p_start_symbol != nullptr) {
        Py_INCREF(p_start_symbol); // needed because PyList_SetItem hands ownership to p_list, see https://docs.python.org/3/extending/extending.html#ownership-rules
        PyList_SetItem(p_list, static_cast<Py_ssize_t>(0), p_start_symbol);
    }

    PyObject* p_weights = PyObject_CallOneArg(query_func, p_list);
    if (!PyList_CheckExact(p_weights))
        throw std::runtime_error("Something went wrong, the Network did not return a list. What happened?");

    static const int RESPONSE_SIZE = static_cast<int>(PyList_Size(p_weights));
    vector<float> res(RESPONSE_SIZE);
    for (int i = 0; i < RESPONSE_SIZE; ++i) {
        PyObject* resp = PyList_GET_ITEM(p_weights, static_cast<Py_ssize_t>(i));
        res[i] = static_cast<float>(PyFloat_AsDouble(resp));
    }

    Py_DECREF(p_list);
    Py_DECREF(p_weights);

    return res;
}

/**
 * @brief Gets a weight distribution along with a hidden representation of the network.
 * 
 * For information on the weight distribution see the get_weight_distribution() method.
 * 
 * @return const std::pair< std::vector<float>, std::vector<float> > <distribution, hidden state>
 */
const std::pair< std::vector<float>, std::vector<float> > 
nn_weighted_output_sul::get_weights_and_state(const std::vector<int>& query_trace, inputdata& id) const {
    static PyObject* p_start_symbol = START_SYMBOL == -1 ? nullptr : PyLong_FromLong(START_SYMBOL);
    static PyObject* p_end_symbol = END_SYMBOL == -1 ? nullptr : PyLong_FromLong(END_SYMBOL);

    PyObject* p_list = p_start_symbol == nullptr ? PyList_New(query_trace.size()) : PyList_New(query_trace.size() + 1);
    int i = p_start_symbol == nullptr ? 0 : 1;
    for (const int flexfringe_symbol : query_trace) {
        int mapped_symbol = stoi(id.get_symbol(flexfringe_symbol));
        PyObject* p_symbol = PyLong_FromLong(mapped_symbol);
        PyList_SetItem(p_list, static_cast<Py_ssize_t>(i), p_symbol);
        ++i;
    }

    if (p_start_symbol != nullptr) {
        Py_INCREF(p_start_symbol); // needed because PyList_SetItem hands ownership to p_list, see https://docs.python.org/3/extending/extending.html#ownership-rules
        PyList_SetItem(p_list, static_cast<Py_ssize_t>(0), p_start_symbol);
    }

    PyObject* p_result = PyObject_CallOneArg(query_func, p_list);
    if (!PyTuple_Check(p_result))
        throw std::runtime_error("Something went wrong, the Network did not return a tuple. What happened?");
    assert(static_cast<int>(PyTuple_Size(p_result)) == 2);

    PyObject* p_weights = PyTuple_GET_ITEM(p_result, static_cast<Py_ssize_t>(0));
    PyObject* p_state = PyTuple_GET_ITEM(p_result, static_cast<Py_ssize_t>(1));

    static const int STATE_SIZE = static_cast<int>(PyList_Size(p_state));    
    vector<float> state(STATE_SIZE);
    for (int i = 0; i < STATE_SIZE; ++i) {
        PyObject* s = PyList_GET_ITEM(p_state, static_cast<Py_ssize_t>(i));
        state[i] = static_cast<float>(PyFloat_AsDouble(s));
    }

    if(PyList_Check(p_weights)){
        // in this branch we get a weight distribution back
        static const int RESPONSE_SIZE = static_cast<int>(PyList_Size(p_weights));
        vector<float> res(RESPONSE_SIZE);
        for (int i = 0; i < RESPONSE_SIZE; ++i) {
            PyObject* resp = PyList_GET_ITEM(p_weights, static_cast<Py_ssize_t>(i));
            res[i] = static_cast<float>(PyFloat_AsDouble(resp));
        }
        return make_pair(res, state);
    }
    else if(PyFloat_Check(p_weights)){
        // binary acceptor model
        vector<float> res(1);
        res[0] = static_cast<float>(PyFloat_AsDouble(p_weights));
        return make_pair(res, state);
    }
    else{
        throw std::runtime_error("Something went wrong, the Network neither returned a float (binary acceptor model\
        , nor did it return a list (language model)). What happened?");
    }
}

/**
 * @brief Destroy the nn sigmoid sul::nn sigmoid sul object
 *
 */
nn_weighted_output_sul::~nn_weighted_output_sul() { Py_Finalize(); }
