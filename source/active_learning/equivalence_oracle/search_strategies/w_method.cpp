/**
 * @file w_method.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-08-18
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "w_method.h"

#include <list>
#include <vector>
#include <cassert>
#include <iostream>

using namespace std;

optional< list<int> > w_method::next(const inputdata& id) {
  return next(id, 0);
}

optional< list<int> > w_method::next(const inputdata& id, const int lower_bound) {
  if(samples_drawn % 10000 == 0) cout << "[" << samples_drawn << "/" << max_samples << "] samples suggested." << endl; 
  if(samples_drawn == max_samples){
    cout << "Exhausted the counterexample search. Wrapping up algorithm." << endl;
    return nullopt;
  } 

  static bool initialized = false;
  if(!initialized){
    const list<int>& alphabet = id.get_alphabet();
    assert(alphabet.size() > 0);

    for(auto s: alphabet) alphabet_vec.push_back(s);
    alphabet_sampler.set_limits(0, alphabet.size()-1);

    initialized = true;
  }

  if(lower_bound > last_lower_bound){
    last_lower_bound = lower_bound;
    length_generator.set_limits(lower_bound, MAX_SEARCH_DEPTH);
  }

  list<int> res;
  const int output_string_length = length_generator.get_random_int();
  for(int i = 0; i < output_string_length; ++i){
    res.push_back(alphabet_vec[alphabet_sampler.get_random_int()]);
  }
  ++samples_drawn;
  return res;
}