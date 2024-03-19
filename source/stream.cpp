// TODO: those are a lot of includes. Do we need them all?
#include "stream.h"
#include <cstdlib>

#include "greedy.h"
#include "state_merger.h"
#include "evaluate.h"
#include "dfasat.h"
#include "evaluation_factory.h"
#include "searcher.h"

#include <vector>
#include <cmath>
#include <string>
#include "parameters.h"
#include "input/abbadingoreader.h"
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <stack>

#include <chrono> // for measuring performance
#include <fstream> // for measuring performance

int P_PERCENT = 0; // to get each i-th percentage. We print the model then

void stream_object::greedyrun_no_undo(state_merger* merger, const int seq_nr, const bool last_sequence, const int n_runs){
    static int count_batches = 0;
    static int count_printouts = 1;

    refinement* top_ref;
    top_ref = merger->get_best_refinement();
    while(top_ref != 0){
      top_ref->doref(merger);
      top_ref = merger->get_best_refinement();
    }

    // print out the state machine before undoing all the merges
    ++count_batches;

    if(seq_nr > count_printouts * P_PERCENT || last_sequence){
      std::cout << "Processed " << count_printouts * 10 << " percent" << std::endl;
      ++count_printouts;
    }
}

/**
 * @brief Runs the stream mode.
 * 
 * @param merger The selected state merger instance.
 * @param param The global parameters.
 * @param input_stream The input data file.
 * @param id Input-data wrapper object. 
 * @return int 
 */
int stream_object::stream_mode(state_merger* merger, std::ifstream& input_stream, abbadingo_inputdata* id) {
    const int BATCH_SIZE = 500;
    unsigned int seq_nr = 0;
    bool last_sequence = false;

    P_PERCENT = static_cast<int>(id->get_max_sequences() / 10); // to track the percent
    std::cout << "P_PERCENT: " << P_PERCENT << std::endl;

    std::ofstream time_doc("times_per_batch.txt");
    //merger->eval->initialize(merger);

    // for performance measurement
    unsigned int n_runs = 0;
    //const int max_runs = 200;
    auto start_time = std::chrono::system_clock::now();

    while(true) {
      //merger->reset(); // TODO: should we use this?
      int read_lines = 0;
      auto batch_start_time = std::chrono::system_clock::now();
      while (read_lines < BATCH_SIZE){
        if(input_stream.eof()){
          last_sequence = true;
          break; // TODO: for experiments, delete afterwards
        }

        ++read_lines;
        ++seq_nr;

        trace* new_trace = mem_store::create_trace();
        id->read_abbadingo_sequence(input_stream, new_trace);

        if(new_trace == nullptr){
          last_sequence = true;
          break; // TODO: for experiments, delete afterwards
        }

        new_trace->sequence = seq_nr;
        id->add_trace_to_apta(new_trace, merger->get_aut());
        if(!ADD_TAILS) new_trace->erase();
      }

      greedyrun_no_undo(merger, seq_nr, last_sequence, n_runs);

      time_doc.flush();
      ++(this->batch_number);
      ++n_runs;


      if(input_stream.eof()){
        std::cout << "Finished parsing file. End of program." << std::endl;
        time_doc.close();
        return 0;
      }
    }
    return 0;
}
