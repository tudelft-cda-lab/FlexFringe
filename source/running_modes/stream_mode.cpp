/**
 * @file stream_mode.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "stream_mode.h"
#include "state_merger.h"
#include "evaluate.h"
#include "refinement_selection_strategies.h"
#include "common.h"
#include "csv.hpp"
#include "output_manager.h"

#include "input/parsers/csvparser.h"
#include "input/parsers/abbadingoparser.h"
#include "input/abbadingoreader.h"

#include <vector>
#include <cmath>
#include <string>
#include "parameters.h"
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <stack>
#include <optional>

#include <chrono>
#include <fstream>

// TODO: throw out those two flags, make them input parameters as well.
const bool RETRY_MERGES = true;
const bool EXPERIMENTAL_RUN = true;

using namespace std;

void stream_mode::initialize() {
  // nothing to do here
}

refinement* stream_mode::determine_next_refinement(state_merger* merger){
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

/**
 * @brief Streaming framework as described in "Learning state machines via efficient hashing of future traces",
 * Baumgartner and Verwer, LearnAUT 2022
 */
void stream_mode::greedyrun_no_undo(state_merger* merger){
    refinement* top_ref;
    top_ref = determine_next_refinement(merger);
    while(top_ref != 0){
      top_ref->doref(merger);
      top_ref = merger->get_best_refinement();
    }

    // TODO: delete the top_refs
}

/**
 * @brief Insert the states of the refinements into the states_to_append_to()
 * set.
 * 
 * @param ref The refinement whose states we want to remember.
 */
void stream_mode::remember_state(refinement* ref){
  this->states_to_append_to.insert(ref->red->get_number());
  
  if(ref->type() == refinement_type::merge_rf_type){
    this->states_to_append_to.insert(dynamic_cast<merge_refinement*>(ref)->blue->get_number());
  }
}

/**
 * @brief Streaming framework as described in "Learning state machines from data streams: A generic strategy and an improved heuristic",
 * Baumgartner and Verwer, ICGI 2023
 */
void stream_mode::greedyrun_retry_merges(state_merger* merger){
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
int stream_mode::run() {
  ifstream input_stream(INPUT_FILE);
  if(!input_stream) {
      LOG_S(ERROR) << "Input file not found, aborting";
      std::cerr << "Input file not found, aborting" << std::endl;
      exit(-1);
  } else {
      std::cout << "Using input file: " << INPUT_FILE << std::endl;
  }

  bool read_csv = false;
  if(INPUT_FILE.compare(INPUT_FILE.length() - 4, INPUT_FILE.length(), ".csv") == 0){
    read_csv = true;
  }

  parser* input_parser;
  if(read_csv) {
    input_parser = new csv_parser(input_stream, csv::CSVFormat().trim({' '}));
  } else {
    input_parser = new abbadingoparser(input_stream);
  }

  eval->initialize_before_adding_traces();
    const int THIS_BATCH_SIZE = STREAMING_BATCH_SIZE;
    
    unsigned int seq_nr = 0;
    bool last_sequence = false;
    while(true) {
      int n_seq_batch = 0;
      while (n_seq_batch < THIS_BATCH_SIZE){
        
        if(input_stream.eof())
          cout << "Reached end of input stream. Waiting for new data." << endl;

        while(input_stream.eof())
          this_thread::sleep_for(chrono::seconds(2));

        ++n_seq_batch;
        ++seq_nr;
        
        std::optional<trace*> trace_opt = id.read_trace(*input_parser, *parser_strategy);

        if(EXPERIMENTAL_RUN && !trace_opt){
          last_sequence = true;
          break;
        }
        else{
          while(!trace_opt) trace_opt = id.read_trace(*input_parser, *parser_strategy); // making sure we got a value; for real streaming
        }
        trace* new_trace = trace_opt.value();
        new_trace->sequence = seq_nr;

        id.add_trace_to_apta(new_trace, merger->get_aut(), true, &(this->states_to_append_to));
        if(!ADD_TAILS) new_trace->erase();
      }

      n_seq_batch = 0;

      if(RETRY_MERGES) greedyrun_retry_merges(merger);
      else greedyrun_no_undo(merger);
      ++(this->batch_number);

      if(DEBUGGING)
        output_manager::print_current_automaton(merger, "Batch_", to_string(this->batch_number));

/*       if(input_stream.eof()){
        if(RETRY_MERGES){
          // one more step, because we undid refinements earlier
          greedyrun_retry_merges(merger); // TODO: is this one needed?

          for(auto top_ref: *currentrun){
              top_ref->doref(merger);
          }
        }

        std::cout << "Finished parsing file. End of program." << std::endl;
        return EXIT_SUCCESS;
      } */
    }
    
    delete input_parser;

    return EXIT_SUCCESS;
}
