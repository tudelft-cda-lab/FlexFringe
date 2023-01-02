#ifndef _STREAM_
#define _STREAM_

#include "refinement.h"
#include "state_merger.h"
#include "input/abbadingoreader.h"

#include <sstream>

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

public:
  stream_object(){
    batch_number = 0;

    currentrun = new refinement_list();
    nextrun = new refinement_list();
  }

  int stream_mode(state_merger* merger, ifstream& input_stream, abbadingo_inputdata* id);
  void greedyrun_no_undo(state_merger* merger, const int seq_nr, const bool last_sequence, const int n_runs);
};

#endif