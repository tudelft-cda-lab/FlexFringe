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
 * @brief Gets the hidden representation from the parameters given. Saves duplicate code. 
 * 
 * @param p_result The result as returned by the network. p_results assumed to be a flattened out matrix, 
 * i.e. a vector that we have to infer the shape from again via HIDDEN_STATE_SIZE.
 * @param offset The initial offset where we find HIDDEN_STATE_SIZE.
 * @return vector< vector<float> > The hidden representations, one per input symbol. Usually including <SOS> and <EOS>.
 */
vector< vector<float> > nn_weighted_output_sul::compile_hidden_rep(PyObject* p_result, const int offset) const {
    
    static const int HIDDEN_STATE_SIZE = static_cast<int>(PyLong_AsLong(PyList_GetItem(p_result, static_cast<Py_ssize_t>(offset)))); // get first list, then return its length 
    const int n_sequences = static_cast<int>( (static_cast<int>(PyList_Size(p_result)) -2) / HIDDEN_STATE_SIZE);
    vector< vector<float> > representations(n_sequences);
    for (int i = 0; i < n_sequences; ++i) {
        vector<float> hidden_rep(HIDDEN_STATE_SIZE);

        for(int j=0; j<HIDDEN_STATE_SIZE; ++j){
            int idx = i * HIDDEN_STATE_SIZE + j + offset + 1; // + offset + 1 because the first elements of p_result are predicted type, and eventually a confidence
            PyObject* s = PyList_GET_ITEM(p_result, static_cast<Py_ssize_t>(idx));
            hidden_rep[j] = static_cast<float>(PyFloat_AsDouble(s));
        }

        representations[i] = move(hidden_rep);
    }

    return representations;
}

/**
 * @brief Gets the type, assumed to be of Sigmoid output and a threshold of 0.5, along with 
 * a hidden representation of the network state.
 * 
 * @return const std::pair< int, std::vector<float> > <the type, the hidden representations for each symbol of sequence>
 */
const std::pair< int, std::vector< std::vector<float> > > 
nn_weighted_output_sul::get_type_and_states(const std::vector<int>& query_trace, inputdata& id) const {
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

    PyObject* p_result = PyObject_CallOneArg(query_func, p_list);
    if (!PyList_Check(p_result))
        throw std::runtime_error("Something went wrong, the Network did not return a list. What happened?");

    // by convention, python script must return a list. list[0]=prediction, list[1]=embedding_dim, rest is hidden_representations 1D
    PyObject* p_type = PyList_GetItem(p_result, static_cast<Py_ssize_t>(0));
    if(!PyLong_Check(p_type)){
        cerr << "Problem with type as returned by Python script. Is it a proper int?" << endl;
        throw exception(); // force the catch block
    }
    else if(!PyLong_CheckExact(p_type)){
        cerr << "Something weird happend here." << endl;
        throw exception();
    }

    int type = static_cast<int>(PyLong_AsLong(p_type));
    vector< vector<float> > representations = compile_hidden_rep(p_result, 1);

    return make_pair(type, representations);
}

/**
 * @brief Gets the type and a confidence value along with a hidden representation of the network state.
 * 
 * @return const std::tuple< int, float, std::vector<float> > <the type, confidence, the hidden representations for each symbol of sequence>
 */
const std::tuple< int, float, std::vector< std::vector<float> > > 
nn_weighted_output_sul::get_type_confidence_and_states(const std::vector<int>& query_trace, inputdata& id) const {
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

    PyObject* p_result = PyObject_CallOneArg(query_func, p_list);
    if (!PyList_Check(p_result))
        throw std::runtime_error("Something went wrong, the Network did not return a list. What happened?");

    // by convention, python script must return a list. list[1]=prediction, list[0]=confidence_in_prediction, list[2]=embedding_dim, rest is hidden_representations 1D
    PyObject* p_confidence = PyList_GetItem(p_result, static_cast<Py_ssize_t>(0));
    if(!PyFloat_Check(p_confidence)){
        cerr << "Problem with type as returned by Python script. Is it a proper float?" << endl;
        throw exception(); // force the catch block
    }
    else if(!PyFloat_CheckExact(p_confidence)){
        cerr << "Something weird happend here." << endl;
        throw exception();
    }
    
    PyObject* p_type = PyList_GetItem(p_result, static_cast<Py_ssize_t>(1));
    if(!PyLong_Check(p_type)){
        cerr << "Problem with type as returned by Python script. Is it a proper int?" << endl;
        throw exception(); // force the catch block
    }
    else if(!PyLong_CheckExact(p_type)){
        cerr << "Something weird happend here." << endl;
        throw exception();
    }

    int type = static_cast<int>(PyLong_AsLong(p_type));
    float confidence = static_cast<float>(PyFloat_AsDouble(p_confidence));
    vector< vector<float> > representations = compile_hidden_rep(p_result, 2);

    return make_tuple(type, confidence, representations);
}

/**
 * @brief Initialize the types to 0 and 1. Which is which depends on how the network was trained.
 *
 */
void nn_weighted_output_sul::init_types() const {
    inputdata_locator::get()->add_type(std::string("Type_0"));
    inputdata_locator::get()->add_type(std::string("Type_1"));
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
