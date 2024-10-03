// TODO: those are a lot of includes. Do we need them all?
#include "stream.h"
#include <cstdlib>

#include "greedy.h"
#include "state_merger.h"
#include "evaluate.h"
#include "dfasat.h"
#include "evaluation_factory.h"
#include "searcher.h"
#include "refinement_selection_strategies.h"

#include "main_helpers.h" // TODO: only for debugging

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

using namespace std;

refinement* stream_object::determine_next_refinement(state_merger* merger){
  if(!DO_ACTIVE_LEARNING) return merger->get_best_refinement();

  static bool initialized = false;
  static shared_ptr<database_sul> database_connector;
  static unique_ptr<evidence_based_strategy> selection_strategy;

  if(!initialized){
    database_connector = make_shared<database_sul>();
    selection_strategy = make_unique<evidence_based_strategy>(database_connector, merger);
    initialized = true;
  }

  shared_ptr<node_to_refinement_map_T> node_to_ref_map_opt = make_shared<node_to_refinement_map_T>();
  auto possible_refs = merger->get_possible_refinements(node_to_ref_map_opt); // TODO: be careful about no refinements possible here and check length of your map first
  if(possible_refs->empty()){
    //cout << "Got a nullptr" << endl;
    return nullptr;
  } 
  
  refinement* res;
  if(possible_refs->size() == 1){
    //cout << "Identified" << endl;
    res = *(possible_refs->begin());
  }
  else{
    //cout << "Run the strategy" << endl;
    res = selection_strategy->perform(possible_refs, node_to_ref_map_opt);
  }

  delete possible_refs;
  return res;
}

void stream_object::greedyrun_no_undo(state_merger* merger, const int seq_nr, const bool last_sequence, const int n_runs){
    refinement* top_ref;
    top_ref = determine_next_refinement(merger);
    while(top_ref != 0){
      top_ref->doref(merger);
      top_ref = determine_next_refinement(merger);
    }
}

/**
 * @brief Insert the states of the refinements into the states_to_append_to()
 * set.
 * 
 * @param ref The refinement whose states we want to remember.
 */
void stream_object::remember_state(refinement* ref){
  this->states_to_append_to.insert(ref->red->get_number());
  
  if(ref->type() == refinement_type::merge_rf_type){
    this->states_to_append_to.insert(dynamic_cast<merge_refinement*>(ref)->blue->get_number());
  }
}

void print_current_automaton_stream(state_merger* merger, const string& output_file, const string& append_string){
    if (OUTPUT_TYPE == "dot" || OUTPUT_TYPE == "both") {
        merger->print_dot(output_file + append_string + ".dot");
    }
    if (OUTPUT_TYPE == "json" || OUTPUT_TYPE == "both") {
        merger->print_json(output_file + append_string + ".json");
    }
    if(OUTPUT_SINKS && !PRINT_WHITE){
        bool red_undo = PRINT_RED;
        PRINT_RED = false;
        bool white_undo = PRINT_WHITE;
        PRINT_WHITE= true;
        bool blue_undo = PRINT_BLUE;
        PRINT_BLUE = true;
        if (OUTPUT_TYPE == "dot" || OUTPUT_TYPE == "both") {
            merger->print_dot(output_file + append_string + "sinks.dot");
        }
        if (OUTPUT_TYPE == "json" || OUTPUT_TYPE == "both") {
            merger->print_json(output_file + append_string + "sinks.json");
        }
        PRINT_RED = red_undo;
        PRINT_WHITE = white_undo;
        PRINT_BLUE = blue_undo;
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

    while(top_ref != nullptr){
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
            top_ref = nullptr;
        }
        else{
            top_ref = currentrun->front();
            currentrun->pop_front();
        }
    }

    top_ref = determine_next_refinement(merger);
    while(top_ref != nullptr){
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

    ++count_printouts;
    if(count_printouts % 100 == 0  || last_sequence){
      cout << "Processed " << count_printouts << " batches" << endl;
    }

    if(count_printouts % 1000 == 0){
      print_current_automaton_stream(merger, "model_batch_nr_", to_string(count_printouts));
    }

    // undo the merges
    while(!refinement_stack.empty()){
      top_ref = refinement_stack.top();
      top_ref->undo(merger);
      refinement_stack.pop();

      remember_state(top_ref);
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

    if(DO_ACTIVE_LEARNING) cout << "Using active learning in conjuntion with streaming." << endl;

    // for performance measurement
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

        //states_to_append_to.insert(merger->get_aut()->get_root()->get_number());
        id->add_trace_to_apta(new_trace, merger->get_aut(), true, &(this->states_to_append_to));
        
        //static int x = 0;
        //print_current_automaton(merger, "automaton_", std::to_string(++x));

        if(!ADD_TAILS) new_trace->erase();
      }

      if(RETRY_MERGES) greedyrun_retry_merges(merger, seq_nr, last_sequence, n_runs);
      else greedyrun_no_undo(merger, seq_nr, last_sequence, n_runs);
      ++(this->batch_number);
      ++n_runs;

      //print_current_automaton(merger, "Batch_", to_string(this->batch_number));

      if(input_stream.eof()){
        if(RETRY_MERGES){
          // one more step, because we undid refinements earlier
          greedyrun_retry_merges(merger, seq_nr, last_sequence, n_runs); // TODO: is this one needed?

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
