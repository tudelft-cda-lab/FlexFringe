/**
 * @file ds_initializer_pre_generate_sequences.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-03-30
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "ds_initializer_pre_generate_sequences.h"


using namespace std;

void ds_initializer_pre_generate_sequences::init(shared_ptr<distinguishing_sequence_fill> ii_handler, unique_ptr<apta>& aut){
  const auto alphabet = aut->get_context()->get_dat()->get_alphabet();
  
  vector< list<int> > collected;
  for(int i=0; i<AL_LONG_TERM_DEPENDENCY_WINSIZE; ++i){
    
  }
}