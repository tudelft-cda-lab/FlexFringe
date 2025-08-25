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

#include <stack>

using namespace std;

/*
 * Recursively build all possible vectors consisting of symbols of alphabet and of 
 * fixed size max_depth.
 */
void ds_initializer_pre_generate_sequences::recurse_layer(/*out*/ vector<int>& current_vector,                                                           
                                                          const int depth,
                                                          const int max_depth,
                                                          const std::vector<int>& alphabet){
  if(depth == max_depth){
    return;
  }
  
  for(const auto symbol: alphabet){
    if(depth == current_vector.size())
        current_vector.push_back(symbol);
    else
        current_vector[depth] = symbol;

    recurse_layer(current_vector, depth+1, max_depth, alphabet);

    if(depth == max_depth-1)
        ii_handler->add_suffix(current_vector);
  }
}

/**
 * @brief Generate all possible sequences with a length of up to AL_LONG_TERM_DEPENDENCY_WINSIZE.
 * As a side effect, those are not real distinguishing sequences anymore, but we are sure to cover all 
 * until the pre-set depth. This enables us to run statistical methods on some SULs as well. 
 */
void ds_initializer_pre_generate_sequences::init(shared_ptr<distinguishing_sequences_handler> ii_handler, unique_ptr<apta>& aut){
  this->ii_handler = ii_handler;
  
  const vector<int>& alphabet = aut->get_context()->get_dat()->get_alphabet();
  if(alphabet.size() == 0){
    throw runtime_error("ERROR: Alphabet invalid");
  }

  for(int max_depth=1; max_depth<=AL_LONG_TERM_DEPENDENCY_WINSIZE; ++max_depth){
    vector<int> current_vector;
    recurse_layer(current_vector, 0, max_depth, alphabet);
  }
  
  cout << "Distinguishing sequences initialized with " << ii_handler->size() << " pre-generated sequences" << endl;
}