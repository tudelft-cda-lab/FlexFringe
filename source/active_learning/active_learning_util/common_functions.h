/**
 * @file common_functions.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-02-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _COMMON_FUNCTIONS_H_
#define _COMMON_FUNCTIONS_H_

#include "evaluate.h"
#include "parameters.h"
#include "apta.h"
#include "tail.h"
#include "trace.h"
#include "refinement.h"
#include "definitions.h"
#include "count_types.h"
#include "inputdata.h"
#include "base_teacher.h"

#include <list>
#include <functional>
#include <utility>
#include <unordered_map>

namespace active_learning_namespace{

  apta_node* get_child_node(apta_node* n, tail* t);
  bool aut_accepts_trace(trace* tr, apta* aut); 
  bool aut_accepts_trace(trace* tr, apta* aut, const count_driven* const eval); 

  const int predict_type_from_trace(trace* tr, apta* aut, inputdata& id);
  trace* concatenate_traces(trace* tr1, trace* tr2);

  /* inline */ void reset_apta(state_merger* merger, const std::list<refinement*>& refs);
  /* inline */ void minimize_apta(list<refinement*>& refs, state_merger* merger);

  std::vector<int> concatenate_strings(const std::vector<int>& pref1, const std::vector<int>& pref2);

  /* __attribute__((always_inline)) */
  /* inline */ trace* vector_to_trace(const std::vector<int>& vec, inputdata& id, const int trace_type = 0);

  void add_sequence_to_trace(/*out*/ trace* new_trace, const std::vector<int> sequence);
  void update_tail(/*out*/ tail* t, const int symbol);

  const double get_probability(trace* tr, inputdata& id, const std::unique_ptr<base_teacher>& teacher, apta* aut,
                                std::shared_ptr< std::unordered_map<apta_node*, std::unordered_map<int, double> > >&  node_response_map);


  /**
   * @brief Compares reference-wrappers of a type.
   * 
   */
  template<typename T>
  struct ref_wrapper_comparator{
  public:
    ref_wrapper_comparator() = default;

    constexpr bool operator()(const std::reference_wrapper<T>& left, const std::reference_wrapper<T>& right ) const { 
      return left.get() < right.get();
    }
  };

  /**
   * @brief For debugging
   */
  template <class it_T>
  [[maybe_unused]]
  void print_sequence(it_T begin, it_T end){
    cout << "seq: ";
    for (; begin != end; ++begin)
        cout << *begin << " ";
    cout << endl;
  }

  [[maybe_unused]]
  void print_list(const std::list<int>& l);

  void print_vector(const std::vector<int>& l);
}

#endif