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

#include <csignal>
#include <vector>
#include <ctime>
#include <unistd.h>
#include "ea_utils.h"

using namespace std;

const bool RETRY_MERGES = true; // if true use new streaming scheme, else use the old one
bool INTERRUPTED = false;

void signal_handler_stream(int signum) {
    if (signum == SIGINT) {
      INTERRUPTED = true;
    }
}

void logMessageStream(const string& message) {
    std::ofstream log("/tmp/flexfringe_log.txt", std::ios::app);
    log << message + "\n";
    log.close();
}


void stream_object::greedyrun_no_undo(state_merger* merger){
    refinement* top_ref;
    top_ref = merger->get_best_refinement();
    while(top_ref != 0){
      top_ref->doref(merger);
      top_ref = merger->get_best_refinement();
    }
}

void print_current_automaton_stream(state_merger* merger, const string& output_file, const string& append_string){
    if (OUTPUT_TYPE == "dot" || OUTPUT_TYPE == "both") {
        merger->print_dot(OUTPUT_DIRECTORY + output_file + append_string + ".dot");
    }
    if (OUTPUT_TYPE == "json" || OUTPUT_TYPE == "both") {
        merger->print_json(OUTPUT_DIRECTORY + output_file + append_string + ".json");
    }
    if(OUTPUT_SINKS && !PRINT_WHITE){
        bool red_undo = PRINT_RED;
        PRINT_RED = false;
        bool white_undo = PRINT_WHITE;
        PRINT_WHITE= true;
        bool blue_undo = PRINT_BLUE;
        PRINT_BLUE = true;
        if (OUTPUT_TYPE == "dot" || OUTPUT_TYPE == "both") {
            merger->print_dot(OUTPUT_DIRECTORY + output_file + append_string + "sinks.dot");
        }
        if (OUTPUT_TYPE == "json" || OUTPUT_TYPE == "both") {
            merger->print_json(OUTPUT_DIRECTORY + output_file + append_string + "sinks.json");
        }
        PRINT_RED = red_undo;
        PRINT_WHITE = white_undo;
        PRINT_BLUE = blue_undo;
    }
}

void stream_object::greedyrun_retry_merges(state_merger* merger){
    queue<refinement*> failed_refs;
    refinement* top_ref;
    this->states_to_append_to.clear();

    if(currentrun->empty()){
      top_ref = merger->get_best_refinement();
    }
    else{
        top_ref = currentrun->front();
        currentrun->pop_front();
    }

    while(top_ref != 0){
        if(top_ref->test_ref_structural(merger)){
          if(top_ref->test_ref_consistency(merger)){
            nextrun->push_back(top_ref);
            current_ref_stack.push(top_ref);
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

    top_ref = merger->get_best_refinement();
    while(top_ref != 0){
      nextrun->push_back(top_ref);
      current_ref_stack.push(top_ref);
      top_ref->doref(merger);

      queue<refinement*> tmp;
      while(!failed_refs.empty()){
        top_ref = failed_refs.front();
        failed_refs.pop();
        if(top_ref->test_ref_structural(merger)){
          if(top_ref->test_ref_consistency(merger)){
            nextrun->push_back(top_ref);
            current_ref_stack.push(top_ref);
            top_ref->doref(merger);
          }
        }
        else{
          tmp.push(top_ref);
        }
      }
      failed_refs = tmp;

      top_ref = merger->get_best_refinement();
    }
}


void stream_object::greedyrun_undo_merges(state_merger* merger){
   // undo the merges that were done the last run (merges done in "greedyrun_retry_merges")
    while(!current_ref_stack.empty()){
      refinement* top_ref = current_ref_stack.top();
      top_ref->undo(merger);
      current_ref_stack.pop();

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
 * @brief Runs the stream mode with a batch of data. The batch consists of all traces read from the input file.
 * 
 * This is the original version of the streaming function, where we can toggle the old and the new version.
 * 
 * @param merger The selected state merger instance.
 * @param param The global parameters.
 * @param input_stream The input data file.
 * @param id Input-data wrapper object. 
 */
 void stream_object::stream_mode_batch(state_merger* merger, std::list<trace*> traces, int trace_batch_nr) {
    // inputdata* id = merger->get_dat();
    // logMessageStream("Starting to process batch of traces.");
    for (auto tr: traces) {
      merger->get_dat()->add_trace_to_apta(tr, merger->get_aut());
    }

    // logMessageStream("Running greedy algorithm with retries.");
    greedyrun_retry_merges(merger);
    // logMessageStream("Finished running greedy algorithm with retries.");
    // logMessageStream("Printing the current automaton to file.");
    // if (trace_batch_nr % 30 == 0) {
    //   print_current_automaton_stream(merger, "model_batch_nr_", to_string(trace_batch_nr));
    // }
    greedyrun_undo_merges(merger);
    logMessageStream("Finished processing trace batch nr: " + to_string(trace_batch_nr));
}


std::vector<apta_node*> stream_object::get_state_sequence_from_trace(state_merger* merger, trace* trace){
    std::vector<apta_node*> state_sequence;
    apta_node* current_state = merger->get_aut()->get_root();
    state_sequence.push_back(current_state);
    tail* t = trace->head;
    while(!t->is_final()){
        // state_sequence.push_back(current_state->guard(t->get_symbol())->get_target());
        current_state = current_state->get_child(t->get_symbol());
        state_sequence.push_back(current_state);
        t = t->future();
    }
    return state_sequence;
}


std::vector<std::vector<apta_node*>> stream_object::get_state_sequences(std::list<trace*> traces, state_merger* merger){
    std::vector<std::vector<apta_node*>> state_sequences;
    refinement_list* ref_list = get_current_run();

    // first redo the refinements to get the correct state sequences
    if (ref_list->size() > 0) {  
      for (auto ref : *ref_list) {
          ref->doref(merger);
      }
    }

    for(auto tr : traces){
        state_sequences.push_back(get_state_sequence_from_trace(merger, tr));
    }

    if (ref_list->size() > 0) {
      // undo the refinements to reset the automaton for new data
      for (auto ref = ref_list->rbegin(); ref != ref_list->rend(); ++ref) {
        (*ref)->undo(merger);
      }
    }

    return state_sequences;
}