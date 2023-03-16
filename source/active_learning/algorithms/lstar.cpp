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

using namespace std;
using namespace active_learning_namespace;

const bool PROCESS_NEGATIVE_TRACES = false; // TODO: refactor this one later
const bool PRINT_ALL_MODELS = false;

lstar_algorithm::lstar_algorithm(const vector<int>& alphabet) : obs_table(observation_table(alphabet)){
  //merger = unique_ptr(state_merger(inputdata*, evaluation_function*, apta*));
}

stack<refinement*> lstar_algorithm::construct_automaton_from_table(unique_ptr<state_merger>& merger, inputdata& id) const {
  //static int sequence_nr = 0;

  const auto& upper_table = obs_table.get_upper_table();
  const auto& lower_table = obs_table.get_lower_table();
  const auto& column_names = obs_table.get_column_names();

  // We iterate over all prefixes and suffixes. TODO: Can cause duplicates? Optimize later
  for(auto row_it = upper_table.cbegin(); row_it != upper_table.cend(); ++row_it){
    const vector<int>& prefix = row_it->first;
    const auto entry = row_it->second;

    //for(auto col_it = column_names.cbegin(); col_it != column_names.cend(); ++col_it){
    for(auto col_it = entry.cbegin(); col_it != entry.cend(); ++col_it){
      const vector<int>& suffix = col_it->first;

      const auto whole_prefix = concatenate_strings(prefix, suffix);
      const auto answer = obs_table.get_answer(prefix, suffix);

/*       cout << "Here comes a trace: ";
      print_vector(whole_prefix); */

/*       int type;
      if(answer==knowledge_t::accepting){
        type = 1;
      }
      else if (answer==knowledge_t::rejecting){
        type = 0;
      }
      else{
        throw logic_error("The table in L* at this point should always be closed.");
      } */

/*       trace* new_trace = mem_store::create_trace(&id);
      new_trace->type = type;
      add_sequence_to_trace(new_trace, clean_prefix);
      
      new_trace->sequence = ++sequence_nr;
      new_trace->finalize(); */

      trace* new_trace = vector_to_trace(whole_prefix, id, answer);

      cout << "whole prefix";
      print_vector(whole_prefix);
      cout << "trace: " << new_trace->to_string() << endl;

      id.add_trace_to_apta(new_trace, merger->get_aut(), set<int>());
    }
  }

  cout << "Building a model => starting a greedy minimization routine" << endl;
  stack<refinement*> refs = minimize_apta(merger.get());

  // For debugging
  if(PRINT_ALL_MODELS){ 
    static int model_nr = 0;
    cout << "Printing model nr. " << model_nr << endl;
    print_current_automaton(merger.get(), "model", "." + to_string(model_nr) + ".not_final");
    ++model_nr;
  }

  return refs;
}

void lstar_algorithm::run_l_star(){
  bool terminated = false;
  int n_runs = 0;

  if(ENSEMBLE_RUNS <= 0){
    cout << "WARNING: runs parameter set to " << ENSEMBLE_RUNS << ". This can cause the algorithm to run indefinitely." << endl;
  }

  // TODO: make those dynamic later
  input_file_sul sul; // TODO: make these generic when you can
  base_teacher teacher(&sul); // TODO: make these generic when you can
  input_file_oracle oracle(&sul); // TODO: make these generic when you can

  inputdata id;
  if(sul.has_input_file()){
    sul.parse_input(id);
  }
  
  auto eval = unique_ptr<evaluation_function>(get_evaluation());
  auto the_apta = unique_ptr<apta>(new apta());
  auto merger = unique_ptr<state_merger>(new state_merger(&id, eval.get(), the_apta.get()));

  while(true){
    cout << "\nIteration: " << n_runs << endl;

    const auto& rows_to_close = vector<pref_suf_t>(obs_table.get_incomplete_rows()); // need a copy, since we're modifying structure in mark_row_complete()
    const auto& column_names = obs_table.get_column_names();
    
    // fill the table until known
    for(const auto& current_row : rows_to_close){
      for(const auto& current_column: column_names){
        if(obs_table.has_record(current_row, current_column)) continue;

        const knowledge_t answer = teacher.ask_membership_query(current_row, current_column);
        obs_table.insert_record(current_row, current_column, answer);
      }
      obs_table.mark_row_complete(current_row);
    }

    //cout << "Added some traces to table. Print: " << endl;
    //obs_table.print();

    if(obs_table.is_closed()){
      static int model_nr = 0;
      stack< refinement* > refs = construct_automaton_from_table(merger, id);
      print_current_automaton(merger.get(), /* OUTPUT_FILE */ "model.", to_string(++model_nr) + ".final"); // printing the final model each time
      cout << "Model nr " << model_nr << endl;

      optional< vector<int> > query_result = oracle.equivalence_query(merger.get());
      if(!query_result){
        cout << "Found consistent automaton => Print." << endl;
        break;
      }
      else{
        // found a counterexample
        cout << "Found counterexample. Resetting apta." << endl;
        reset_apta(merger.get(), refs);
        const vector<int>& cex = query_result.value();
        obs_table.extent_columns(cex);

        cout << "CEX: ";
        print_vector(cex);
        print_current_automaton(merger.get(), /* OUTPUT_FILE */ "model.", to_string(model_nr) + ".after_undo"); // printing the final model each time
      }
    }
    else{
      obs_table.extend_lower_table(); // extending the lower table, rerun
    }

    ++n_runs;
    if(ENSEMBLE_RUNS > 0 && n_runs == ENSEMBLE_RUNS) break;
  }

/*   if(ENSEMBLE_RUNS > 0 && n_runs == ENSEMBLE_RUNS) cout << "Reached maximum number of iterations. Printing model" << endl;
  print_current_automaton(merger.get(), OUTPUT_FILE, ".final"); */
}