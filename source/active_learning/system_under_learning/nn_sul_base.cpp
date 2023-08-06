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

#include <sstream>
#include <iostream>
#include <map> // TODO: make this one and the one in r_alphabet unordered

using namespace std;

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

  stringstream cmd;
  cmd << "sys.path.append( os.path.join(os.getcwd(), ";
  cmd << PYTHON_SCRIPT_PATH;
  cmd << "))";
  PyRun_SimpleString(cmd.str().c_str());

  // load the module
  PyObject* pName = PyUnicode_FromString(PROGRAM_PATH);
  if(pName == NULL){
    Py_DECREF(pName);
    cerr << "Error in retrieving the name string of the program path. Terminating program." << endl;
    exit(1);
  }

  pModule = PyImport_Import(pName);
  if(pModule == NULL){
    Py_DECREF(pName);
    Py_DECREF(pModule);
    cerr << "Error in loading python module " << PYTHON_MODULE_NAME << ". Terminating program." << endl;
    exit(1);
  }

  query_func = PyObject_GetAttrString(pModule, "do_query");
  if(query_func == NULL || !PyCallable_Check(query_func)){
    Py_DECREF(pName);
    Py_DECREF(pModule);
    Py_DECREF(query_func);
    cerr << "Problem in loading the query function. Terminating program." << endl;
    exit(1);
  }

  alphabet_func = PyObject_GetAttrString(pModule, "get_alphabet");
  if(alphabet_func == NULL || !PyCallable_Check(alphabet_func)){
    Py_DECREF(pName);
    Py_DECREF(pModule);
    Py_DECREF(query_func);
    Py_DECREF(alphabet_func);
    cerr << "Problem in loading the get_alphabet function. Terminating program." << endl;
    exit(1);
  }

  // TODO: set the alphabet here
  PyObject* p_alphabet = PyObject_CallObject(alphabet_func, NULL);
  if(!PyDict_Check(p_alphabet)){
    Py_DECREF(pName);
    Py_DECREF(pModule);
    Py_DECREF(query_func);
    Py_DECREF(alphabet_func);
    cerr << "get_alphabet() function in python script did not return a dictionary-type." << endl;
    exit(1);
  }

  map<string, int> mapped_alphabet;
  PyObject* p_keys = PyDict_Keys(pValue);
  const auto size = static_cast<int>(PyList_Size(p_keys));
  for(int i = 0; i < size; ++i){
    PyObject* p_key = PyList_GetItem(p_keys, static_cast<Py_ssize_t>(i));
    PyObject* p_value = PyDict_GetItem(pValue, pleft);

    std::string left;
    int right;
    try{
      left = PyUnicode_AsUTF8(PyObject_Str(pleft));// pyUnicode_fromString(pleft);
      right = PyLong_AsLong(pright);
    }
    catch{
      Py_DECREF(pName);
      Py_DECREF(pModule);
      Py_DECREF(query_func);
      Py_DECREF(alphabet_func);
      Py_DECREF(p_key);
      Py_DECREF(p_value);
      cerr << "Alphabet dict returned by get_alphabet() must be a string->int dictionary." << endl;
      exit(1);
    }

    mapped_alphabet[std::move(left)] = std::move(right);
  }

  id.set_r_alphabet(std::move(mapped_alphabet));

  Py_INCREF(pName);
  Py_INCREF(pModule);
  Py_INCREF(query_func);
  Py_INCREF(alphabet_func);

  cout << "Python module " << PYTHON_MODULE_NAME << " loaded and initialized successfully." << endl;
}