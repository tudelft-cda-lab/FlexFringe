#ifndef _STREAM_
#define _STREAM_

#include "refinement.h"
#include "state_merger.h"
#include "input/abbadingoreader.h"
#include "parameters.h"
#include "input/parsers/reader_strategy.h"
#include "active_learning/system_under_learning/database_sul.h"

#include <sstream>
#include <stack>
#include <unordered_set>
#include <memory>

/**
 * @brief Class to realize the streaming mode, when observing a stream of data, e.g. network data.
 *
 */
class stream_object{

private:
  int batch_number; // TODO: naming
  refinement_list* currentrun;
  refinement_list* nextrun;

  std::unordered_set<int> states_to_append_to; // keeping track of states that we can append to with ease
  reader_strategy* parser_strategy;

  __attribute__((always_inline))
  inline void greedyrun_no_undo(state_merger* merger, const int seq_nr, const bool last_sequence, const int n_runs);

  __attribute__((always_inline))
  inline void greedyrun_retry_merges(state_merger* merger, const int seq_nr, const bool last_sequence, const int n_runs); // for experiments

  __attribute__((always_inline))
  inline refinement* determine_next_refinement(state_merger* merger);

  __attribute__((always_inline))
  inline void remember_state(refinement* ref);

public:
  stream_object(){
    batch_number = 0;

    // TODO: make those smart pointers
    currentrun = new refinement_list();
    nextrun = new refinement_list();

    parser_strategy = new in_order();
  }

  ~stream_object(){
    delete currentrun;
    delete nextrun;
    delete parser_strategy;
  }

  //__attribute__((flatten)) // inlines all subsequent functions into this one, (potentially) increases speed at cost of larger code and compile time
  int stream_mode(state_merger* merger, std::ifstream& input_stream, inputdata* id, parser* input_parser);
};

#endif
