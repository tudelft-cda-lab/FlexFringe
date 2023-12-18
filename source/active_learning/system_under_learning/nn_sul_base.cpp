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
#include <string>

using namespace std;

/**
 * @brief Inserts element into vector, raises exception if it didn't work.
 * 
 * @param pylist The vector
 * @param item The item
 * @param idx The index
 */
void nn_sul_base::set_list_item(PyObject* p_list, PyObject* p_item, const int idx) const {
  int r = PyList_SetItem(p_list, idx, p_item);
  if(r==-1){
    cerr << "Error when setting items in python-vector." << endl;  
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
    pos = static_cast<int>(INPUT_FILE.find_last_of("\\"));
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
  //PyRun_SimpleString("print('Script uses the following interpreter: ', sys.executable)\n");
  //PyRun_SimpleString("print(sys.path)"); // for debugging

  //PyRun_SimpleString("import mlflow");

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
    cerr << "Error in loading python module " << INPUT_FILE << ". Terminating program.\n \
Possible hints for debugging: Does your interpreter have access to the imported libraries? \
Is your sys.path environment correct? Perhaps try to import the libraries as a standalone and \
see what happens." << endl;
    exit(1);
  }

  query_func = PyObject_GetAttrString(p_module, "do_query");
  if(query_func == NULL || !PyCallable_Check(query_func)){
    Py_DECREF(p_name);
    Py_DECREF(p_module);
    cerr << "Problem in loading the query function. Terminating program." << endl;
    exit(1);
  }

  PyObject* p_model_path = PyUnicode_FromString(APTA_FILE.c_str());
  if(p_model_path == NULL){
    Py_DECREF(p_name);
    Py_DECREF(p_module);
    Py_DECREF(query_func);
    cerr << "p_model_path could not be set correctly." << endl;
    exit(1);
  }

  load_model_func = PyObject_GetAttrString(p_module, "load_nn_model");
  if(load_model_func == NULL || !PyCallable_Check(load_model_func)){
    Py_DECREF(p_name);
    Py_DECREF(p_module);
    Py_DECREF(query_func);
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
    Py_DECREF(p_model_path);
    cerr << "Unexpected return from calling the load_model-function. Function should not return value. Did an exception happen?" << endl;
    exit(1);
  }

  alphabet_func = PyObject_GetAttrString(p_module, "get_alphabet");
  if(alphabet_func == NULL || !PyCallable_Check(alphabet_func)){
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
  if(p_alphabet == NULL || !PyList_Check(p_alphabet)){
    Py_DECREF(p_name);
    Py_DECREF(p_module);
    Py_DECREF(query_func);
    Py_DECREF(alphabet_func);
    Py_DECREF(load_model_func);
    Py_DECREF(p_model_path);
    cerr << "get_alphabet() function in python script did not return a list-type. Is \
    the path to the aptafile correct and contains the alphabet? Alphabet has correct name \
    as in python script?" << endl;
    exit(1);
  }

  cout << "Setting internal flexfringe alphabet, inferred from the network's training alphabet" << endl;
  vector<int> input_alphabet;
  const auto size = static_cast<int>(PyList_Size(p_alphabet));
  const int start_symbol = START_SYMBOL; const int end_symbol = END_SYMBOL;
  for(int i = 0; i < size; ++i){
    PyObject* p_item = PyList_GetItem(p_alphabet, static_cast<Py_ssize_t>(i));

    int item;
    try{
      item = PyLong_AsLong(p_item);
    }
    catch(...){
      Py_DECREF(p_name);
      Py_DECREF(p_module);
      Py_DECREF(query_func);
      Py_DECREF(alphabet_func);
      Py_DECREF(load_model_func);
      Py_DECREF(p_model_path);
      Py_DECREF(p_item);
      cerr << "Alphabet returned by Python script must be a vector of integer-values." << endl;
      exit(1);
    }

    Py_DECREF(p_item);
    if(item != start_symbol && item != end_symbol) input_alphabet.push_back(std::move(item));
  }

  id.set_alphabet(input_alphabet);

  Py_INCREF(p_name);
  Py_INCREF(p_module);
  Py_INCREF(query_func);
  Py_INCREF(alphabet_func);

  cout << "Python module " << INPUT_FILE << " loaded and initialized successfully." << endl;
  init_types();
}