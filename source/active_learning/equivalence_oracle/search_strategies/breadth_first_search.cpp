/**
 * @file breadth_first_search.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-04-13
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "breadth_first_search.h"

#include <stack>
#include <list>
#include <utility>
#include <cassert>
#include <iostream>

using namespace std;

/**
 * @brief Normal BFS through the alphabet, testing for size. I implemented this using two stacks. This requires us 
 * lots of copying. An alternative method would be to instead implement recursively, which comes at the cost of the heap, but that does 
 * not matter much if we assume the max depth to be not too extreme.
 * 
 * @param sul The SUl.
 * @param id The inputdata.
 * @return optional< const list<int> > A counterexample if found.
 */
optional< list<int> > bfs_strategy::next(const inputdata& id) {
  if(depth == BFS_MAX_DEPTH) return nullopt;

  static const list<int> alphabet = id.get_alphabet();
  assert(alphabet.size() > 0);
  static list<int>::const_iterator alphabet_it = alphabet.begin();

  [[unlikely]]
  if(depth == 0){
    old_search.push( list<int>{*alphabet_it} );
    
    if(alphabet_it == alphabet.end()){
      cout << "Depth: " << depth << endl;

      ++depth;
      alphabet_it = alphabet.begin();
      return old_search.top();
    }

    ++alphabet_it;
    return old_search.top();
  }

  // Flow: take from old_search, append and put on current_search.
  const auto& prefix = old_search.top();
  auto new_pref = list<int>(prefix);
  new_pref.push_back(*alphabet_it);
  curr_search.push( std::move(new_pref) );

  if(alphabet_it == alphabet.end()){
    alphabet_it = alphabet.begin();
    old_search.pop();

    if(old_search.empty()){
      cout << "Depth: " << depth << endl;

      ++depth;
      old_search = std::move(curr_search);
      curr_search = stack< list<int> >();
      return old_search.top();
    }
    
    return curr_search.top();
  } 

  ++alphabet_it;
  return curr_search.top();
}
