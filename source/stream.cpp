// TODO: those are a lot of includes. Do we need them all?
#include "stream.h"
#include <cstdlib>

#include "greedy.h"
#include "state_merger.h"
#include "evaluate.h"
#include "dfasat.h"
#include "evaluation_factory.h"
#include "searcher.h"
#include "parameters.h"

#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <stack>
#include <optional>

#include <chrono> // for measuring performance
#include <fstream> // for measuring performance

const bool RETRY_MERGES = true;
const bool EXPERIMENTAL_RUN = true;
const bool DO_ACTIVE_LEARNING = false;

int P_PERCENT = 0; // to get each i-th percentage. We print the model then

refinement* stream_object::determine_next_refinement(state_merger* merger){
  if(!DO_ACTIVE_LEARNING) return merger->get_best_refinement();

  optional<node_to_refinement_map_T> node_to_ref_map_opt;
  auto possible_refs = merger->get_possible_refinements(node_to_ref_map_opt); // TODO: be careful about no refinements possible here and check length of your map first
  if(possible_refs->empty()) return nullptr; 
  else if(possible_refs->size() == 1){
    refinement* res = *rs->begin();
    delete possible_refs;
    return res; 
  }

  // TODO: Do the strategy on the refinements here
  

  delete possible_refs;
}

void stream_object::greedyrun_no_undo(state_merger* merger, const int seq_nr, const bool last_sequence, const int n_runs){

    refinement* top_ref;
    top_ref = determine_next_refinement(merger);
    while(top_ref != 0){
      top_ref->doref(merger);
      top_ref = determine_next_refinement(merger);
    }
}

void stream_object::greedyrun_retry_merges(state_merger* merger, const int seq_nr, const bool last_sequence, const int n_runs){
    static int count_printouts = 1;
    static int c = 0;

    stack< refinement* > refinement_stack;
    queue<refinement*> failed_refs;
    refinement* top_ref;

    this->states_to_append_to.clear();

    if(currentrun->empty()){
      top_ref = determine_next_refinement(merger);
    }
    else{
        top_ref = currentrun->front();
        currentrun->pop_front();
    }

    while(top_ref != 0){
        if(top_ref->test_ref_structural(merger)){
          if(top_ref->test_ref_consistency(merger)){
            nextrun->push_back(top_ref);
            refinement_stack.push(top_ref);
            top_ref->doref(merger);
          }
        }
        else{
          failed_refs.push(top_ref);
        }

        if(currentrun->empty()){
            top_ref = 0;
        }
        else{
            top_ref = currentrun->front();
            currentrun->pop_front();
        }
    }

    top_ref = determine_next_refinement(merger);
    while(top_ref != 0){
      nextrun->push_back(top_ref);
      refinement_stack.push(top_ref);
      top_ref->doref(merger);

      queue<refinement*> tmp;
      while(!failed_refs.empty()){
        top_ref = failed_refs.front();
        failed_refs.pop();
        if(top_ref->test_ref_structural(merger)){
          if(top_ref->test_ref_consistency(merger)){
            nextrun->push_back(top_ref);
            refinement_stack.push(top_ref);
            top_ref->doref(merger);
          }
        }
        else{
          tmp.push(top_ref);
        }
      }
      failed_refs = tmp;

      top_ref = determine_next_refinement(merger);
    }

    if(seq_nr > count_printouts * P_PERCENT || last_sequence){
      cout << "Processed " << count_printouts * 10 << " percent" << endl;
      ++count_printouts;
    }

    // undo the merges
    while(!refinement_stack.empty()){
      top_ref = refinement_stack.top();
      top_ref->undo(merger);
      refinement_stack.pop();

      states_to_append_to.insert(top_ref->red->get_number());
      if(top_ref->type() == refinement_type::merge_rf_type){
        states_to_append_to.insert(dynamic_cast<merge_refinement*>(top_ref)->blue->get_number());
      }
    }

    delete currentrun;
    currentrun = nextrun;
    nextrun = new refinement_list();
}

/**
 * @brief Runs the stream mode.
 * 
 * This is the original version of the streaming function, where we can toggle the old and the new version.
 * 
 * @param merger The selected state merger instance.
 * @param param The global parameters.
 * @param input_stream The input data file.
 * @param id Input-data wrapper object. 
 * @return int 
 */
int stream_object::stream_mode(state_merger* merger, ifstream& input_stream, inputdata* id, parser* input_parser) {
    unsigned int seq_nr = 0;
    bool last_sequence = false;

    P_PERCENT = static_cast<int>(id->get_max_sequences() / 10); // to track the percent
    cout << "P_PERCENT: " << P_PERCENT << endl;
    unsigned int n_runs = 0;

    while(true) {
      int read_lines = 0;
      while (read_lines < BATCH_SIZE){
        if(input_stream.eof()){
          last_sequence = true;
          break; // TODO: for experiments, delete afterwards since this will terminate the algortithm, but not necessarily the stream
        }

        ++read_lines;
        ++seq_nr;
        
        std::optional<trace*> trace_opt = id->read_trace(*input_parser, *parser_strategy);

        if(EXPERIMENTAL_RUN && !trace_opt){
          last_sequence = true;
          break;
        }
        else{
          while(!trace_opt) trace_opt = id->read_trace(*input_parser, *parser_strategy); // making sure we got a value; for real streaming
        }
        trace* new_trace = trace_opt.value();
        new_trace->sequence = seq_nr;
        
        id->add_trace_to_apta(new_trace, merger->get_aut(), &(this->states_to_append_to));
        if(!ADD_TAILS) new_trace->erase();
      }

      if(RETRY_MERGES) greedyrun_retry_merges(merger, seq_nr, last_sequence, n_runs);
      else greedyrun_no_undo(merger, seq_nr, last_sequence, n_runs);

      ++(this->batch_number);
      ++n_runs;

      if(input_stream.eof()){
        if(RETRY_MERGES) {
          greedyrun_retry_merges(merger, seq_nr, last_sequence, n_runs);

          for(auto top_ref: *currentrun){
              top_ref->doref(merger);
          }
        }

        cout << "Finished parsing file. End of program." << endl;
        return 0;
      }
    }
    return 0;
}
