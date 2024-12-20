/**
 * @file running_mode_factory.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "running_mode_factory.h"
#include "parameters.h"

#include "active_learning_mode.h"
#include "dfasat_mode.h"
#include "differencing_mode.h"
#include "ensemble_mode.h"
#include "greedy_mode.h"
#include "interactive_mode.h"
#include "predict_mode.h"
#include "regex_mode.h"
#include "search_mode.h"
#include "stream_mode.h"
#include "subgraphextraction_mode.h"

using namespace std;

/**
 * @brief Selects a mode and initializes it.
 */
unique_ptr<running_mode_base> running_mode_factory::get_mode(const string& operation_mode){
  unique_ptr<running_mode_base> mode;

  if(operation_mode == "active_learning"){
    cout << "Active learning mode selected" << endl;
    mode = make_unique<active_learning_mode>();
  }
  else if(operation_mode == "dfasat"){
    cout << "SAT solver mode selected" << endl;
    mode = make_unique<dfasat_mode>();
  }
  else if(operation_mode == "differencing"){
    cout << "Behavioral differencing mode selected" << endl;
    mode = make_unique<differencing_mode>();
  }
  else if(operation_mode == "bagging"){
    cout << "Ensemble mode selected" << endl;
    mode = make_unique<ensemble_mode>();
  }
  else if(operation_mode == "greedy"){
    cout << "Greedy mode selected" << endl;
    mode = make_unique<greedy_mode>();
  }
  else if(operation_mode == "interactive"){
    cout << "Interactive mode selected" << endl;
    mode = make_unique<interactive_mode>();
  }
  else if(operation_mode == "predict"){
    cout << "Predict mode selected" << endl;
    mode = make_unique<predict_mode>();
  }
  else if(operation_mode == "regex"){
    cout << "Regex mode selected" << endl;
    mode = make_unique<regex_mode>();
  }
  else if(operation_mode == "search"){
    cout << "Search mode selected" << endl;
    mode = make_unique<search_mode>();
  }
  else if(operation_mode == "streaming"){
    cout << "Stream mode selected" << endl;
    mode = make_unique<stream_mode>();
  }
  else if(operation_mode == "subgraphextraction"){
    cout << "subgraphextraction mode selected" << endl;
    mode = make_unique<subgraphextraction_mode>(); // TODO: the state merger
  }
  else
    throw std::invalid_argument("unknown operation mode");
  
  mode->initialize();
  return mode;
}

unique_ptr<running_mode_base> running_mode_factory::get_mode(){
  return running_mode_factory::get_mode(OPERATION_MODE);
}
