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
    log << message << std::endl;
    log.close();
}


void stream_object::greedyrun_no_undo(state_merger* merger, const int seq_nr, const bool last_sequence){
    refinement* top_ref;
    top_ref = merger->get_best_refinement();
    while(top_ref != 0){
      top_ref->doref(merger);
      top_ref = merger->get_best_refinement();
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

void stream_object::greedyrun_retry_merges(state_merger* merger, const int seq_nr, const bool last_sequence){
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


void stream_object::greedyrun_undo_merges(state_merger* merger, const int seq_nr, const bool last_sequence){
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
 * @return int 
 */
vector<double> stream_object::stream_mode_batch(state_merger* merger, ifstream& input_stream, parser* input_parser) {
    unsigned int seq_nr = 0;
    bool last_sequence = false;
    inputdata* id = merger->get_dat();;
    this->parser_strategy = new in_order();
    vector<double> fitnesses;

    logMessageStream("Starting to read batch of traces from input file.");

    while (!input_stream.eof()){
      ++seq_nr;
      std::optional<trace*> trace_opt = id->read_trace(*input_parser, *parser_strategy);;
      if(!trace_opt){
        last_sequence = true;
        break;
      }

      trace* new_trace = trace_opt.value();
      logMessageStream("Read trace " + new_trace->to_string() + " from input file.");
      new_trace->sequence = seq_nr;
      this->batch_traces.push_back(new_trace);
      
      id->add_trace_to_apta(new_trace, merger->get_aut());
      // if(!ADD_TAILS) new_trace->erase();
    }


    if(RETRY_MERGES) {
      greedyrun_retry_merges(merger, seq_nr, last_sequence);
      // now we simulate each trace on the automaton and get the sequence of states 
      std::vector<std::vector<apta_node*>> state_sequences;
      for (auto tr : this->batch_traces) {
        state_sequences.push_back(get_state_sequence_from_trace(merger, tr));
      }

      fitnesses = EA_utils::compute_fitnesses(state_sequences, merger->get_aut()->get_root(), FITNESS_TYPE);

      print_current_automaton_stream(merger, "model_batch_nr_", to_string(this->batch_number));
      greedyrun_undo_merges(merger, seq_nr, last_sequence);
    }
    else {
      greedyrun_no_undo(merger, seq_nr, last_sequence);
    }

    ++(this->batch_number);    
    // Clear the traces before reading the next batch.
    for (auto tr : this->batch_traces) {
      tr->erase();
    }

    cout << "Finished parsing batch of traces " << to_string(this->batch_number) << endl;
    return fitnesses;
}

/**
 * @brief Run stream mode with pipe as input
 * 
 * @param merger The selected state merger instance.
 * @param batch_size The size of the batch.
 * @return int 
 */
std::vector<std::pair<double, int>> stream_object::stream_mode(state_merger* merger, int batch_size, int buffer_size) {
    char buffer[buffer_size]; // size of buffer to read from the pipe.
    std::vector<char> line_buffer;
    signal(SIGINT, signal_handler_stream); // setup signal handler to catch SIGINT (Ctrl+C)
    time_t last_read_time = time(nullptr); // record the last time something was read from pipe. 
    ssize_t bytes_read;
    std::vector<std::string> current_batch; 
    int sequence_number = 0;

    std::vector<pair<double, int>> state_sizes_num_list;

    // Read inputs from pipe. This is done via the STDIN_FILENO file descriptor.
    while (!INTERRUPTED && difftime(time(nullptr), last_read_time) < 5) {
        bytes_read = read(STDIN_FILENO, buffer, buffer_size); // read from the pipe
        if (bytes_read > 0) {
            for (int i = 0; i < bytes_read; ++i) {
                char ch = buffer[i];
                if (ch == '\n') { // If newline character is encountered
                    // Output the collected line
                    // line_buffer.push_back('\0'); // Null-terminate to make it a C-style string
                    std::cout << "Read line: " << std::string(line_buffer.begin(), line_buffer.end()) << std::endl;
                    current_batch.push_back(std::string(line_buffer.begin(), line_buffer.end()));
                    if (current_batch.size() >= batch_size) {
                      
                      // read each trace and add it to the apta.
                      for (const auto& str : current_batch) {
                        inputdata* id = merger->get_dat(); // get the input data object.
                        reader_strategy* parser_strategy = new in_order(); // create a parser strategy.
                        std::istringstream iss(str); // create the input stream.
                        auto parser = abbadingoparser::single_trace(iss); // initialize a parser.
                        trace* new_trace = id->read_trace(parser, *parser_strategy).value(); // read the trace.
                        id->add_trace_to_apta(new_trace, merger->get_aut()); // add the trace to the apta.
                        greedyrun_retry_merges(merger, sequence_number, true); // run the greedy algorithm to do the merges
                        greedyrun_undo_merges(merger, sequence_number, true); // undo the merges.
                        ++sequence_number; // increment the sequence number.
                      }

                      current_batch.clear(); // clear and wait for the next batch of traces.
                    }

                    line_buffer.clear();
                } else {
                    line_buffer.push_back(ch); // Add character to the buffer.
                }
            }
        }
        else {
          continue; // we wait until timeout or until the process is interrupted.
        }
    }

    if (INTERRUPTED) {
        std::cout << "Interrupted by user. Exiting..." << std::endl;
    } else {
        std::cout << "Timeout after 10 minutes. Exiting..." << std::endl;
    }

    return state_sizes_num_list;
}


std::vector<apta_node*> stream_object::get_state_sequence_from_trace(state_merger* merger, trace* trace){
    std::vector<apta_node*> state_sequence;
    apta_node* current_state = merger->get_aut()->get_root();
    state_sequence.push_back(current_state);
    tail* t = trace->head;
    while(!t->is_final()){
        current_state = current_state->get_child(t->get_symbol());
        state_sequence.push_back(current_state);
        t = t->future();
    }
    return state_sequence;
}