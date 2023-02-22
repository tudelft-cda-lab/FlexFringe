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
#include "lstar_teacher.h"
//#include "lstar_oracle.h"

#include "parameters.h"
#include "abbadingo_sul.h"

using namespace std;
using namespace active_learning_namespace;

lstar_algorithm::lstar_algorithm(vector<int>& alphabet) : obs_table(observation_table(alphabet)){}

void lstar_algorithm::run_l_star(){
  bool terminated = false;
  int n_runs = 0;

  abbadingo_sul sul; // TODO: update
  lstar_teacher teacher;
  //lstar_oracle oracle;

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
      construct_automaton_from_table();
    }
    else{
      obs_table.extend_lower_table(); // extending the lower table, rerun
    }

    ++n_runs;
  }
}