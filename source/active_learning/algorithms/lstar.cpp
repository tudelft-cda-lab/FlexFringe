/**
 * @file lstar.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-02-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "lstar.h"
#include "base_teacher.h"
#include "input_file_sul.h"
#include "common_functions.h"
#include "input_file_oracle.h"

#include "state_merger.h"
#include "parameters.h"
#include "inputdata.h"
#include "mem_store.h"
#include "greedy.h"
#include "main_helpers.h"

#include <stdexcept>
#include <iostream>
#include <optional>
#include <utility>

using namespace std;
using namespace active_learning_namespace;

const bool PRINT_ALL_MODELS = false; // for debugging

/**
 * @brief Does what it says it does.
 * 
 * @param obs_table 
 * @param merger 
 * @param id 
 * @return stack<refinement*> 
 */
const list<refinement*> lstar_algorithm::construct_automaton_from_table(const observation_table& obs_table, unique_ptr<state_merger>& merger, inputdata& id) const {
  const auto& upper_table = obs_table.get_upper_table();
  const auto& lower_table = obs_table.get_lower_table();
  const auto& column_names = obs_table.get_column_names();

  // We iterate over all prefixes and suffixes. TODO: Can cause duplicates? Optimize later
  for(auto row_it = upper_table.cbegin(); row_it != upper_table.cend(); ++row_it){
    const list<int>& prefix = row_it->first;
    const auto entry = row_it->second;

    //for(auto col_it = column_names.cbegin(); col_it != column_names.cend(); ++col_it){
    for(auto col_it = entry.cbegin(); col_it != entry.cend(); ++col_it){
      const list<int>& suffix = col_it->first;

      const auto whole_prefix = concatenate_strings(prefix, suffix);
      const auto answer = obs_table.get_answer(prefix, suffix);

      trace* new_trace = vector_to_trace(whole_prefix, id, answer);
      id.add_trace_to_apta(new_trace, merger->get_aut(), set<int>());
    }
  }

  cout << "Building a model => starting a greedy minimization routine" << endl;
  const list<refinement*> refs = minimize_apta(merger.get());

  return refs;
}

/**
 * @brief This algorithms main method.
 * 
 * @param id The inputdata, already initialized with input file.
 */
void lstar_algorithm::run(inputdata& id){
  bool terminated = false;
  int n_runs = 0;

  observation_table obs_table(id.get_alphabet());

  sul->pre(id);
  
  auto eval = unique_ptr<evaluation_function>(get_evaluation());
  auto the_apta = unique_ptr<apta>(new apta());
  auto merger = unique_ptr<state_merger>(new state_merger(&id, eval.get(), the_apta.get()));

  list< refinement* > refs; // we keep track of refinements
  while(ENSEMBLE_RUNS > 0 && n_runs < ENSEMBLE_RUNS){
    cout << "\nIteration: " << n_runs << endl;

    const auto& rows_to_close = list<pref_suf_t>(obs_table.get_incomplete_rows()); // need a copy, since we're modifying structure in mark_row_complete(). 
    const auto& column_names = obs_table.get_column_names();
    
    // fill the table until known
    for(const auto& current_row : rows_to_close){
      for(const auto& current_column: column_names){
        if(obs_table.has_record(current_row, current_column)) continue;

        const int answer = teacher->ask_membership_query(current_row, current_column, id);
        obs_table.insert_record(current_row, current_column, answer);
      }
      obs_table.mark_row_complete(current_row);
    }

    if(obs_table.is_closed()){
      refs = construct_automaton_from_table(obs_table, merger, id);

      if(PRINT_ALL_MODELS){
        static int model_nr = 0;
        print_current_automaton(merger.get(),"model.", to_string(++model_nr) + ".not_final"); // printing the final model each time
        cout << "Model nr " << model_nr << endl;
      }

      optional< pair< list<int>, int > > query_result = oracle->equivalence_query(merger.get(), teacher);
      if(!query_result){
        cout << "Found consistent automaton => Print." << endl;
        print_current_automaton(merger.get(), OUTPUT_FILE, ".final"); // printing the final model each time
        return;
      }
      else{
        const list<int>& cex = query_result.value().first;
        const int type = query_result.value().second;
        auto cex_tr = vector_to_trace(cex, id, type);

        reset_apta(merger.get(), refs);
        obs_table.extent_columns(cex);

        if(PRINT_ALL_MODELS){
          static int model_nr = 0;
          print_current_automaton(merger.get(),"model.", to_string(++model_nr) + ".after_undo"); // printing the final model each time
          cout << "Model nr " << model_nr << endl;
        }
      }
    }
    else{
      obs_table.extend_lower_table(); // extending the lower table, rerun
    }

    ++n_runs;
    if(ENSEMBLE_RUNS > 0 && n_runs == ENSEMBLE_RUNS){
      cout << "Maximum of runs reached. Printing automaton." << endl;
      for(auto top_ref: refs){
        top_ref->doref(merger.get());
      }
      print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
    }
  }
}