/**
 * @file ds_initializer_collect_from_apta.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-03-30
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "ds_initializer_collect_from_apta.h"

#include <stack>
#include <iostream>

#include "parameters.h"
 
using namespace std;

void ds_initializer_collect_from_apta::init(shared_ptr<distinguishing_sequences_handler> ii_handler, unique_ptr<apta>& aut){
  cout << "Collecting some distinguishing sequences from the initial apta" << endl;
  const int MAX_INIT_DEPTH = 2;
  int count = 0;

  stack<apta_node*> l_stack;
  stack<apta_node*> r_stack;

  l_stack.push(aut->get_root());
  while(!l_stack.empty()){
    apta_node* left = l_stack.top();
    l_stack.pop();

    cout << ++count << ",";
    cout.flush();

    r_stack.push(aut->get_root());
    while(!r_stack.empty()){
      apta_node* right = r_stack.top();
      r_stack.pop();

      ii_handler->pre_compute(aut, left, right); // collects the sequences

      if(right->get_depth() < MAX_INIT_DEPTH){
        for(apta_node* n_child : right->get_unmerged_child_nodes())
          r_stack.push(n_child);
      }
    }

    if(left->get_depth() < MAX_INIT_DEPTH){
      for(apta_node* n_child : left->get_unmerged_child_nodes())
        l_stack.push(n_child);
    }
  }

  cout << endl;
}