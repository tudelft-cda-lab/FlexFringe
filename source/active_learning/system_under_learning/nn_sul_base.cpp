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

#include <sstream>
#include <iostream>
#include <map> // TODO: make this one and the one in r_alphabet unordered
#include <stdexcept>

using namespace std;

/**
 * @brief Inserts element into list, raises exception if it didn't work.
 * 
 * @param pylist The list
 * @param item The item
 * @param idx The index
 */
void nn_sul_base::set_list_item(PyObject* p_list, PyObject* p_item, const int idx) const {
  int r = PyList_SetItem(p_list, idx, p_item);
  if(r==-1){
    cerr << "Error when setting items in python-list." << endl;  
    throw bad_alloc();
  } 
}

/**
 * @brief Initializes the python environment and 
 * enables us to load the python module.
 * 
 * @param id Not needed.
 */
void nn_sul_base::pre(inputdata& id){  
  Py_Initialize(); // start python interpreter
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("import os");

  // set the path accordingly
  std::string PYTHON_SCRIPT_PATH; std::string PYTHON_MODULE_NAME; int pos;
  #if defined(_WIN32)
    pos = static_cast<int>(APAFILE.find_last_of("\\"));
  #else 
    pos = static_cast<int>(INPUT_FILE.find_last_of("/"));
  #endif
  PYTHON_SCRIPT_PATH = INPUT_FILE.substr(0, pos);
  PYTHON_MODULE_NAME = INPUT_FILE.substr(pos+1, INPUT_FILE.size()-pos-4); // -pos-4 to get rid off the .py ending

  stringstream cmd;
  cmd << "sys.path.append( os.path.join(os.getcwd(), \"";
  cmd << PYTHON_SCRIPT_PATH;
  cmd << "\") )";

  PyRun_SimpleString(cmd.str().c_str());
  //PyRun_SimpleString("print(sys.path)"); // useful for debugging

  // load the module
  PyObject* p_name = PyUnicode_FromString(PYTHON_MODULE_NAME.c_str());
  if(p_name == NULL){
    cerr << "Error in retrieving the name string of the program path. Terminating program." << endl;
    exit(1);
  }

  cout << "Loading module: " << PyUnicode_AsUTF8(PyObject_Str(p_name)) << endl;

  p_module = PyImport_Import(p_name);
  if(p_module == NULL){
    Py_DECREF(p_name);
    cerr << "Error in loading python module " << INPUT_FILE << ". Terminating program." << endl;
    exit(1);
  }

  query_func = PyObject_GetAttrString(p_module, "do_query");
  if(query_func == NULL || !PyCallable_Check(query_func)){
    Py_DECREF(p_name);
    Py_DECREF(p_module);
    cerr << "Problem in loading the query function. Terminating program." << endl;
    exit(1);
  }

  alphabet_func = PyObject_GetAttrString(p_module, "get_alphabet");
  if(alphabet_func == NULL || !PyCallable_Check(alphabet_func)){
    Py_DECREF(p_name);
    Py_DECREF(p_module);
    Py_DECREF(query_func);
    cerr << "Problem in loading the get_alphabet function. Terminating program." << endl;
    exit(1);
  }

  // TODO: set the alphabet here
  PyObject* p_model_path = PyUnicode_FromString(APTA_FILE.c_str());
  if(p_model_path == NULL){
    Py_DECREF(p_name);
    Py_DECREF(p_module);
    Py_DECREF(query_func);
    Py_DECREF(alphabet_func);
    cerr << "p_model_path could not be set correctly." << endl;
    exit(1);
  }

  cout << "Loading alphabet in python module." << endl;
  PyObject* p_alphabet = PyObject_CallOneArg(alphabet_func, p_model_path);
  if(p_alphabet == NULL || !PyDict_Check(p_alphabet)){
    Py_DECREF(p_name);
    Py_DECREF(p_module);
    Py_DECREF(query_func);
    Py_DECREF(alphabet_func);
    cerr << "get_alphabet() function in python script did not return a dictionary-type. Is \
    the path to the aptafile correct and contains the alphabet? Alphabet has correct name \
    as in python script?" << endl;
    exit(1);
  }

  load_model_func = PyObject_GetAttrString(p_module, "load_nn_model");
  if(load_model_func == NULL || !PyCallable_Check(load_model_func)){
    Py_DECREF(p_name);
    Py_DECREF(p_module);
    Py_DECREF(query_func);
    Py_DECREF(alphabet_func);
    Py_DECREF(p_model_path);
    cerr << "Problem in loading the load_model function. Terminating program." << endl;
    exit(1);
  }

  cout << "Loading the Neural network model in python module." << endl;
  PyObject* p_load_return = PyObject_CallOneArg(load_model_func, p_model_path);
  if(p_load_return != Py_None){
    Py_DECREF(p_name);
    Py_DECREF(p_module);
    Py_DECREF(query_func);
    Py_DECREF(alphabet_func);
    Py_DECREF(p_model_path);
    cerr << "Unexpected return from calling the load_model-function. Function should not return value. Did an exception happen?" << endl;
    exit(1);
  }

  cout << "Setting internal flexfringe alphabet, inferred from the network's training alphabet" << endl;
  map<string, int> mapped_alphabet;
  PyObject* p_keys = PyDict_Keys(p_alphabet);
  const auto size = static_cast<int>(PyList_Size(p_keys));
  for(int i = 0; i < size; ++i){
    PyObject* p_key = PyList_GetItem(p_keys, static_cast<Py_ssize_t>(i));
    PyObject* p_value = PyDict_GetItem(p_alphabet, p_key);

    std::string key;
    int value;
    try{
      key = PyUnicode_AsUTF8(PyObject_Str(p_key));// pyUnicode_fromString(pleft);
      value = PyLong_AsLong(p_value);
    }
    catch(...){
      Py_DECREF(p_name);
      Py_DECREF(p_module);
      Py_DECREF(query_func);
      Py_DECREF(alphabet_func);
      //Py_DECREF(p_key); // TODO: this is not very correct, I need to check individually
      //Py_DECREF(p_value); // TODO: this is not very correct, I need to check individually
      cerr << "Alphabet dict returned by get_alphabet() must be a string->int dictionary." << endl;
      exit(1);
    }

    mapped_alphabet[std::move(key)] = std::move(value);
  }

  id.set_alphabet(std::move(mapped_alphabet));

  Py_INCREF(p_name);
  Py_INCREF(p_module);
  Py_INCREF(query_func);
  Py_INCREF(alphabet_func);

  cout << "Python module " << INPUT_FILE << " loaded and initialized successfully." << endl;
}