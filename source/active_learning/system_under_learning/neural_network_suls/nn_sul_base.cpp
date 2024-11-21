/**
 * @file nn_sul_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Documentation of CPython-API: https://docs.python.org/3/c-api/index.html
 * @version 0.1
 * @date 2023-08-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "nn_sul_base.h"
#include "parameters.h"
#include "inputdatalocator.h"

#include <iostream>
#include <map> // TODO: make this one and the one in r_alphabet unordered
#include <sstream>
#include <stdexcept>
#include <string>

using namespace std;

#ifdef __FLEXFRINGE_PYTHON

/**
 * @brief Generic function that prints an error when the output of the python script is not as expected.
 * 
 */
void nn_sul_base::print_p_error() const {
    std::cerr << "Something went wrong in the Python script, see line below. Terminating program" << std::endl;
    PyErr_Print();
    exit(EXIT_FAILURE);
}

/**
 * @brief Initialize the types to 0 and 1. Which is which depends on how the network was trained.
 * 
 * TODO: Is there a better way to deal with this?
 *
 */
void nn_sul_base::init_types() const {
    inputdata_locator::get()->add_type(std::string("Type_0"));
    inputdata_locator::get()->add_type(std::string("Type_1"));
}

/**
 * @brief Sometimes when expecting discrete values we can either receive integers or strings from 
 * the Python script, where strings are more likely. This function checks the value it received, and 
 * returns either the directly converted integer value in case an integer was received, or the 
 * flexfringe mapped internal integer representation in case a string has been received. The mapping
 * is done according to the alphabet/r_alphabet twin structure in the inputdata object.
 */
int nn_sul_base::pyobj_to_int(PyObject* p_obj, inputdata& id) const{
    int res;

    // first check on string: This is the hot path
    if(PyUnicode_CheckExact(p_obj)){
      int type = id.get_reverse_type(PyUnicode_AsUTF8(p_obj));
      if(type > id.get_alphabet_size()){
          id.add_type(PyUnicode_AsUTF8(p_obj));
      }
    }
    else if(PyLong_Check(p_obj)){
      type = PyLong_AsLong(p_obj);
    }
    else{
        cerr << "Problem with type as returned by Python script. Type must be string or int value." << endl;
        throw exception();
    }

    return res;
}

/**
 * @brief Like strings_to_pylist, but it first converts c_list into a vector 
 * with string representations using the internal inputdata structure as a mapping.
 * 
 * WARNING: If p_list_out already has elements, then we create a memory leak here.
 * 
 * @param p_list_out The Python list to write into.
 * @param c_list The C++ vector of strings.
 */
void nn_sul_base::input_sequence_to_pylist(PyObject* p_list_out, const vector<int>& c_list) const {
    static inputdata* id = inputdata_locator::get();

    for(int i = 0; i < c_list.size(); ++i){
        PyObject* p_symbol = PyUnicode_FromString(id->get_symbol(c_list[i]).c_str());
        PyList_SET_ITEM(p_list_out, i, p_symbol);
    }
}

/**
 * @brief Initializes the python environment and
 * enables us to load the python module.
 *
 * @param id Not needed.
 */
void nn_sul_base::pre(inputdata& id) {
    string CONNECTOR_SCRIPT = APTA_FILE2.size() == 0 ? INPUT_FILE : APTA_FILE2;
    if (CONNECTOR_SCRIPT.compare(CONNECTOR_SCRIPT.length() - 3, CONNECTOR_SCRIPT.length(), ".py") != 0)
        throw invalid_argument("The connector script for the Python module is not ending on .py. Please check your input again.");

    Py_Initialize(); // start python interpreter
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("import os");

    // set the path accordingly
    std::string PYTHON_SCRIPT_PATH;
    std::string PYTHON_MODULE_NAME;
    int pos;
#if defined(_WIN32)
    pos = static_cast<int>(CONNECTOR_FILE.find_last_of("\\"));
#else
    pos = static_cast<int>(CONNECTOR_FILE.find_last_of("/"));
#endif
    PYTHON_SCRIPT_PATH = CONNECTOR_FILE.substr(0, pos);
    PYTHON_MODULE_NAME =
        CONNECTOR_FILE.substr(pos + 1, CONNECTOR_FILE.size() - pos - 4); // -pos-4 to get rid off the .py ending

    stringstream cmd;
    cmd << "sys.path.append( os.path.join(os.getcwd(), \"";
    cmd << PYTHON_SCRIPT_PATH;
    cmd << "\") )\n";
    cmd << "sys.path.append( os.path.join(os.getcwd(), \"source/active_learning/system_under_learning/python/util\") )";
    PyRun_SimpleString(cmd.str().c_str());
    // for debugging
    // PyRun_SimpleString("print('Script uses the following interpreter: ', sys.executable)\n");
    //PyRun_SimpleString("print(sys.path)"); 

    // load the module
    PyObject* p_name = PyUnicode_FromString(PYTHON_MODULE_NAME.c_str());
    if (p_name == NULL) {
        cerr << "Error in retrieving the name string of the program path. Terminating program." << endl;
        exit(1);
    }

    cout << "Loading module: " << PyUnicode_AsUTF8(PyObject_Str(p_name)) << endl;

    p_module = PyImport_Import(p_name);
    if (p_module == NULL) {
        Py_DECREF(p_name);
        cerr << "Error in loading python module " << INPUT_FILE << ". Terminating program.\n \
Possible hints for debugging: Does your interpreter have access to the imported libraries? \
Is your sys.path environment correct? Perhaps try to import the libraries as a standalone and \
see what happens."
             << endl;
        exit(1);
    }

    query_func = PyObject_GetAttrString(p_module, "do_query");
    if (query_func == NULL || !PyCallable_Check(query_func)) {
        Py_DECREF(p_name);
        Py_DECREF(p_module);
        cerr << "Problem in loading the query function. Terminating program." << endl;
        exit(1);
    }

    p_model_path = PyUnicode_FromString(APTA_FILE.c_str());
    if (p_model_path == NULL) {
        Py_DECREF(p_name);
        Py_DECREF(p_module);
        Py_DECREF(query_func);
        cerr << "p_model_path could not be set correctly." << endl;
        exit(1);
    }

    load_model_func = PyObject_GetAttrString(p_module, "load_nn_model");
    if (load_model_func == NULL || !PyCallable_Check(load_model_func)) {
        Py_DECREF(p_name);
        Py_DECREF(p_module);
        Py_DECREF(query_func);
        Py_DECREF(p_model_path);
        cerr << "Problem in loading the load_model function. Terminating program." << endl;
        exit(1);
    }

    cout << "Loading the Neural network model in python module." << endl;
    PyObject* p_load_return = PyObject_CallOneArg(load_model_func, p_model_path);
    if (p_load_return != Py_None) {
        Py_DECREF(p_name);
        Py_DECREF(p_module);
        Py_DECREF(query_func);
        Py_DECREF(p_model_path);
        cerr << "Unexpected return from calling the load_model-function. Function should not return value. Did an "
                "exception happen?"
             << endl;
        exit(1);
    }

    PyObject* alphabet_func = PyObject_GetAttrString(p_module, "get_alphabet");
    if (alphabet_func == NULL || !PyCallable_Check(alphabet_func)) {
        Py_DECREF(p_name);
        Py_DECREF(p_module);
        Py_DECREF(query_func);
        Py_DECREF(load_model_func);
        Py_DECREF(p_model_path);
        cerr << "Problem in loading the get_alphabet function. Terminating program." << endl;
        exit(1);
    }

    cout << "Loading alphabet in python module." << endl;
    PyObject* p_alphabet = PyObject_CallOneArg(alphabet_func, p_model_path);
    if (p_alphabet == NULL || !PyList_Check(p_alphabet)) {
        Py_DECREF(p_name);
        Py_DECREF(p_module);
        Py_DECREF(query_func);
        Py_DECREF(alphabet_func);
        Py_DECREF(load_model_func);
        Py_DECREF(p_model_path);
        cerr << "get_alphabet() function in python script did not return a list-type. Is \
    the path to the aptafile correct and contains the alphabet? Alphabet has correct name \
    as in python script?"
             << endl;
        exit(1);
    }

    cout << "Setting internal flexfringe alphabet, inferred from the network's training alphabet" << endl;
    vector<string> input_alphabet;
    const auto size = static_cast<int>(PyList_Size(p_alphabet));
    for (int i = 0; i < size; ++i) {
        PyObject* p_item = PyList_GetItem(p_alphabet, static_cast<Py_ssize_t>(i));
        if(!PyUnicode_Check(p_item)){
            Py_DECREF(p_name);
            Py_DECREF(p_module);
            Py_DECREF(query_func);
            Py_DECREF(alphabet_func);
            Py_DECREF(load_model_func);
            Py_DECREF(p_model_path);
            Py_DECREF(p_item);
            cerr << "Alphabet returned by Python script must be a vector of string values." << endl;
            exit(1);
        }
        
        string item;
        try {
            const char* s =  PyUnicode_AsUTF8(p_item);
            item = string(s);
        } catch (...) {
            Py_DECREF(p_name);
            Py_DECREF(p_module);
            Py_DECREF(query_func);
            Py_DECREF(alphabet_func);
            Py_DECREF(load_model_func);
            Py_DECREF(p_model_path);
            Py_DECREF(p_item);
            cerr << "Alphabet returned by Python script must be a vector of string- or integer-values." << endl;
            exit(1);
        }

        // Note: SOS and EOS are not allowed to be returned back
        input_alphabet.push_back(std::move(item));
    }

    id.set_alphabet(input_alphabet);

    Py_DECREF(p_name);
    Py_DECREF(p_alphabet);
    Py_DECREF(alphabet_func);
    Py_DECREF(p_load_return);

    cout << "Python module " << INPUT_FILE << " loaded and initialized successfully." << endl;
    init_types();
}

/**
 * @brief Destroy the nn sul base::nn sul base object
 * 
 * Free py-objects and close the python-interpreter.
 * 
 */
nn_sul_base::~nn_sul_base(){
    Py_DECREF(p_module);
    Py_DECREF(p_model_path);
    Py_DECREF(load_model_func);
    Py_DECREF(query_func);

    Py_Finalize();
}

/* Dummy implementation when Python disabled to get it to compile on platforms without Python Dev Headers. */
#else

void nn_sul_base::pre(inputdata& id) {
    throw std::logic_error("Enable this feature with -DENABLE_PYTHON=ON on cmake.");
}

const sul_response nn_sul_base::do_query(const vector<int>& query_trace, inputdata& id) const {
    throw std::logic_error("Enable this feature with -DENABLE_PYTHON=ON on cmake.");
}

/* void nn_sul_base::init_types() const {
    throw std::logic_error("Enable this feature with -DENABLE_PYTHON=ON on cmake.");
}

void nn_sul_base::print_p_error() const {
    throw std::logic_error("Enable this feature with -DENABLE_PYTHON=ON on cmake.");
} */

#endif /* __FLEXFRINGE_PYTHON */
