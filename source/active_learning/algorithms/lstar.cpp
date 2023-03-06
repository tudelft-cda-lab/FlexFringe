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

lstar_algorithm::lstar_algorithm(vector<int>& alphabet) : obs_table(observation_table(alphabet)){
  //merger = unique_ptr(state_merger(inputdata*, evaluation_function*, apta*));
}

pref_suf_t lstar_algorithm::concatenate_prefixes(const pref_suf_t& pref1, const pref_suf_t& pref2) const {
  pref_suf_t res(pref1);
  res.insert(res.end(), pref2.begin(), pref2.end());
  return res;
}

vector<refinement*> lstar_algorithm::construct_automaton_from_table(unique_ptr<state_merger> merger, intputdata& id) const {
  // step 1: construct traces
  // step 2: sort traces
  // step 3: add traces to apta, step by step

  static int sequence_nr = 0;

  const auto upper_table = obs_table.get_upper_table();
  const auto lower_table = obs_table.get_lower_table();
  const auto column_names = obs_table.get_column_names();
  // We iterate over all prefixes and suffixes. TODO: Can cause duplicates? Optimize later
  for(const auto row_it: upper_table){
    const auto& prefix = row_it.first; // pointer?
    for(const auto suffix: column_names){
      const auto whole_prefix = concatenate_prefixes(prefix, suffix);
      const auto answer = obs_table.get_answer(prefix, suffix);

      int type;
      if(answer==knowledge_t::accepting){
        type = 1;
      }
      else if (answer==knowledge_t::rejecting){
        type = 0;
      }
      else{
        throw logic_error("The table in L* at this point should always be closed.");
      }

      trace* new_trace = mem_store::create_trace(id);
      new_trace->type = type;
      add_sequence_to_trace(new_trace, whole_prefix);
      new_trace->sequence = ++sequence_nr;
      new_trace->finalize();

      id.add_trace_to_apta(new_trace, merger->get_aut(), set<int>());
    }
  }
  vector<refinement*> refs = minimize_apta(merger.get());
  return refs;
}

void lstar_algorithm::run_l_star(){
  bool terminated = false;
  int n_runs = 0;

  input_file_sul sul;
  base_teacher teacher;
  input_file_oracle oracle;

  inputdata id;
  if(sul.has_input_file()){
    auto input_stream = sul.get_input_stream();
    sul.parse_input_file(input_stream, id);
    input_stream.close();
  }
  
  auto eval = unique_ptr<evaluation_function>(get_evaluation());
  auto the_apta = unique_ptr<apta>(new apta());
  auto merger = unique_ptr<state_merger>(new state_merger(&id, evaluation_function*, apta*));

  while(!terminated || (ENSEMBLE_RUNS > 0 && n_runs < ENSEMBLE_RUNS)){
    const auto& rows_to_close = obs_table.get_incomplete_rows();
    const auto& column_names = obs_table.get_column_names();

    // fill the table until known
    for(const auto& current_row : rows_to_close){
      for(const auto& current_column: column_names){
        if(obs_table.has_record(current_row, current_column)){
          continue;
        }

        const knowledge_t answer = teacher.ask_membership_query(current_row, current_column);
        obs_table.insert_record(current_row, current_column, answer);
      }
      obs_table.mark_row_complete(current_row);
    }

    if(obs_table.is_closed()){
      vector< refinement* > refs = construct_automaton_from_table(merger, id);
      optional< vector<int> > query_result = oracle->equivalence_query(merger.get());
      if(!query_result){
        cout << "Found consistent automaton. Print and terminate program." << endl;
        print_current_automaton(merger, OUTPUT_FILE, ".final");
        cout << "Terminate program." << endl;
      }
      else{
        // found a counterexample
        reset_apta(merger.get(), refs);
        const vector<int>& cex = query_result.value();
        obs_table.extend_column_names(cex);
      }
    }
    else{
      obs_table.extend_lower_table(); // extending the lower table, rerun
    }

    ++n_runs;
  }
}