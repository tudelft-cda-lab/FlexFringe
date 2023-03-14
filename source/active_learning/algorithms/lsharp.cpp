/**
 * @file lsharp.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The (strategic) L#-algorithm, as described by Vandraager et al. (2022): "A New Approach for Active Automata Learning Based on Apartness"
 * @version 0.1
 * @date 2023-03-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "lsharp.h"
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

lsharp_algorithm::lsharp_algorithm(const std::vector<int>& alphabet) : alphabet(alphabet) {
  // TODO
}

/**
 * @brief We want the root state to be complete, and it is not a blue state. Hence we want to make sure it is complete at this stage.
 * 
 * @param aut 
 */
void lsharp_algorithm::init_root_state(apta* aut) const {
  // TODO
}

void lsharp_algorithm::proc_counterex(apta* aut, const std::vector<int>& counterex) const {
  // TODO: this is a counterexample strategy
}

void lsharp_algorithm::run_l_sharp(){
  bool terminated = false;
  int n_runs = 0;

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

  while(!terminated || (ENSEMBLE_RUNS > 0 && n_runs < ENSEMBLE_RUNS)){

    ++n_runs;
  }

  if(ENSEMBLE_RUNS > 0 && n_runs == ENSEMBLE_RUNS) cout << "Reached maximum number of iterations. Printing model" << endl;
  print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
}