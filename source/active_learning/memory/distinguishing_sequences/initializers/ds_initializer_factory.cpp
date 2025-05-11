/**
 * @file ds_initializer_factory.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-03-30
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "ds_initializer_factory.h"
#include "ds_initializer_registration.h"

#include "src/ds_initializer_collect_from_apta.h"
#include "src/ds_initializer_pre_generate_sequences.h"

#include <iostream>
#include <stdexcept>

using namespace std;
    
unique_ptr<ds_initializer_base> ds_initializer_factory::get_initializer(string_view name){
  using ds_init_t = ds_initializer_registration::ds_initializers_t;

  if(name.empty()){
    cout << "Empty ii-handler initializer chosen. No initialization will take place." << endl;
    return make_unique<ds_initializer_base>(); 
  }
  else if(name == ds_initializer_registration::get_initializer_name(ds_init_t::collect_from_apta)){
    return make_unique<ds_initializer_collect_from_apta>();
  }
  else if(name == ds_initializer_registration::get_initializer_name(ds_init_t::pre_generate_sequences)){
    return make_unique<ds_initializer_pre_generate_sequences>();
  }
  else{
    throw invalid_argument("II-Initializer name not implemented.");
  }
}


