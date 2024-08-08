#ifndef _STREAM_
#define _STREAM_

#include "refinement.h"
#include "state_merger.h"
#include "input/parsers/reader_strategy.h"
#include "input/parsers/abbadingoparser.h"
#include "input/inputdata.h"

#include <sstream>
#include <stack>

/**
 * @brief Class to realize the streaming mode, when observing a stream of data, e.g. network data.
 *
 */
class stream_object{

private:
  // TODO: remove global objects, make them local
  //int STREAM_COUNT = 0;
  int batch_number; // TODO: naming
  refinement_list* currentrun;
  refinement_list* nextrun;
  std::stack<refinement*> current_ref_stack;
  std::set<int> states_to_append_to; // keeping track of states that we can append to with ease
  reader_strategy* parser_strategy;
  std::list<trace*> batch_traces;

public:
  stream_object(){
    batch_number = 0;

    currentrun = new refinement_list();
    nextrun = new refinement_list();
    reader_strategy* parser_strategy = new in_order();

  }
  
  ~stream_object(){
    delete currentrun;
    delete nextrun;
    delete parser_strategy;
  }


  std::vector<double> stream_mode_batch(state_merger* merger, std::ifstream& input_stream, parser* input_parser);
  std::vector<std::pair<double, int>> stream_mode(state_merger* merger, int batch_size, int buffer_size);
  void greedyrun_no_undo(state_merger* merger, const int seq_nr, const bool last_sequence);
  void greedyrun_retry_merges(state_merger* merger, const int seq_nr, const bool last_sequence); // for experiments
  void greedyrun_undo_merges(state_merger* merger, const int seq_nr, const bool last_sequence); // for experiments
  std::vector<apta_node*> get_state_sequence_from_trace(state_merger* merger, trace* trace);
  int get_batch_number();

};

#endif