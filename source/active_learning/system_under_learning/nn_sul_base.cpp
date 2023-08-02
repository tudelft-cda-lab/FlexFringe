/**
 * @file nn_sul_base.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-08-02
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "nn_sul_base.h"

#include <sstream>
#include <iostream>

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

  cout << "Python module " << PYTHON_MODULE_NAME << " loaded successfully" << endl;
}