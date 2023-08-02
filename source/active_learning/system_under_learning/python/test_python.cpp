/**
 * @file test_nn.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Test file to check the python connector. Tutorial on how to connect C++ and Python: 
 * https://www.codeproject.com/Articles/820116/Embedding-Python-program-in-a-C-Cplusplus-code
 * @version 0.1
 * @date 2023-07-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#define PY_SSIZE_T_CLEAN // recommended, see https://docs.python.org/3/extending/extending.html#a-simple-example
#include <Python.h>

#include <iostream>
#include <string>

using namespace std;

const auto PROGRAM_PATH = "python_test";
//const auto PROGRAM_PATH = "python_test.py";

int main(){
    Py_Initialize();

    // this is so we can find the module
    PyRun_SimpleString("import os");
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(os.getcwd())");

    //auto fp = fopen(PROGRAM_PATH, "r");
    //PyRun_SimpleFile(fp, PROGRAM_PATH + ".py");
    //PyRun_File(fp, PROGRAM_PATH, 0, nullptr, nullptr);

    cout << "Program path: " << PROGRAM_PATH << endl;
    PyObject* pName = PyUnicode_FromString(PROGRAM_PATH);

    auto s = PyUnicode_AsUTF8(PyObject_Str(pName));
    cout << "The name: " << s << endl;

    PyObject* pModule = PyImport_Import(pName);
    if(pModule){
        // test simple function
        cout << "Test with simple return value: " << endl;
 		PyObject* pFunc1 = PyObject_GetAttrString(pModule, "test_no_arg");
		if(pFunc1 && PyCallable_Check(pFunc1)){
			PyObject* pValue = PyObject_CallObject(pFunc1, NULL);
			cout << "Return value with no args: " << PyLong_AsLong(pValue) << endl;
		}
		else{
			cout << "Something went wrong" << endl;
		}

        // test function with global variable
        cout << "Test with global variable: " << endl;
 		PyObject* pFunc2 = PyObject_GetAttrString(pModule, "test_with_global");
		if(pFunc2 && PyCallable_Check(pFunc2)){
			PyObject* pValue = PyObject_CallObject(pFunc2, NULL);
			cout << "Return value with global variable call 1: " << PyLong_AsLong(pValue) << endl;
		}
        if(pFunc2 && PyCallable_Check(pFunc2)){
			PyObject* pValue = PyObject_CallObject(pFunc2, NULL);
			cout << "Return value with global variable call 2: " << PyLong_AsLong(pValue) << endl;
		}
		else{
			cout << "Something went wrong" << endl;
		}
	}
	else
	{
		cout << "ERROR: Module not imported\n" << endl;
	}
    //query_trace(module);

    Py_Finalize();
    return 0;
}


/** executes the module module 
    *
    * @param module: string containing the name of the module
    * @param configuration_map: configuration map matching the module to be launched
    * @param values_list: list of maps containing all the measures
    * @return integer: <0 module error, >0 success, =0 couldn't load module
    *
    * the name of the module shall match configuration file's 
    * datastore item and the name of the python module in the
    * extension's directory
    *
    * the measures are a list of maps having the following nomenclature:
    * "k" : "name of the measure" [string]
    * "u" : "unit of the measure" [string]
    * "v" : "value of the measure" [string containing float]
    * "p" : "precision of the measure" [string containing float]
    */
int query_trace(const std::string& module/* , 
                        string_string_map& configuration_map,
                        std::vector<string_string_map*>& values_list */) {

    PyObject *pName, *pModule, *pDict, *pFunc, *pValue, *pArgs, *pValues=NULL, *pConfig=NULL, *k, *v;
    
    int ret = 0;

#if PY_MAJOR_VERSION >= 3
    pName = PyUnicode_FromString(module.c_str());
#else
    pName = PyString_FromString(module.c_str());
#endif

    cout << Py_TYPE(pName) << endl;


    // Load the module object
    pModule = PyImport_Import(pName);

    if (pModule == NULL) {
        Py_DECREF(pName);
        return 0;
    }

    // pDict is a borrowed reference 
    pDict = PyModule_GetDict(pModule);

    if (pDict == NULL) {
        Py_DECREF(pName);
        Py_DECREF(pModule);
        return 0;
    }

/*     // pFunc is also a borrowed reference 
    pFunc = PyDict_GetItemString(pDict, "push_to_datastore");

    if (pFunc == NULL) {
        Py_DECREF(pName);
        Py_DECREF(pModule);
        Py_DECREF(pDict);
        return 0;
    }


    if (PyCallable_Check(pFunc)) {
        pArgs = PyTuple_New(2);

        // values
        pValues = PyList_New(0);
        for (std::vector<string_string_map*>::iterator it=values_list.begin(); it != values_list.end(); ++it) {
            pValue = PyDict_New();
            for (string_string_map::iterator val_it=(*it)->begin(); val_it != (*it)->end(); ++val_it) {
                k = PyString_FromString(val_it->first.c_str());
                v = PyString_FromString(val_it->second.c_str());
                PyDict_SetItem(pValue, k, v);
            }
            PyList_Append(pValues, pValue);
        }
        pValue = NULL;
        PyTuple_SetItem(pArgs, 0, pValues);

        // config
        pConfig = PyDict_New();
        for (string_string_map::iterator it=configuration_map.begin(); it != configuration_map.end(); ++it) {
            k = PyString_FromString(it->first.c_str());
            v = PyString_FromString(it->second.c_str());
            PyDict_SetItem(pConfig, k, v);
        }
        PyTuple_SetItem(pArgs, 1, pConfig);

        pValue = PyObject_CallObject(pFunc, pArgs);

        if (pArgs != NULL) {
            Py_DECREF(pArgs);
        }

        if (pValue != NULL) {
            ret = PyInt_AsLong(pValue);
            Py_DECREF(pValue);
        } else {
            PyErr_Print();
            ret = 0;
        }
    } */

    return ret;
}