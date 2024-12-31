/**
 * @file stream_mode.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _STREAM_MODE_H_
#define _STREAM_MODE_H_

#include "running_mode_base.h"
#include "state_merger.h"
#include "refinement.h"
#include "input/abbadingoreader.h"
#include "parameters.h"
#include "input/parsers/reader_strategy.h"
#include "active_learning/system_under_learning/database_sul.h"

#include <sstream>
#include <stack>
#include <unordered_set>
#include <memory>

#if defined(_MSC_VER) 
#  define FLEXFRINGE_ALWAYS_INLINE inline
#else
#  define FLEXFRINGE_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

/**
 * @brief Class to realize the streaming mode, when observing a stream of data, e.g. network data.
 *
 */
class stream_mode : public running_mode_base {
  private:
    int batch_number = 0;
    refinement_list* currentrun;
    refinement_list* nextrun;

    std::unordered_set<int> states_to_append_to; // keeping track of states that we can append to with ease
    reader_strategy* parser_strategy;

    FLEXFRINGE_ALWAYS_INLINE void greedyrun_no_undo(state_merger* merger);

    FLEXFRINGE_ALWAYS_INLINE void greedyrun_retry_merges(state_merger* merger); // for experiments

    FLEXFRINGE_ALWAYS_INLINE refinement* determine_next_refinement(state_merger* merger);

    FLEXFRINGE_ALWAYS_INLINE void remember_state(refinement* ref);

  public:
    stream_mode(){
      currentrun = new refinement_list();
      nextrun = new refinement_list();
      parser_strategy = new in_order(); // TODO: more generic? 
    }

    ~stream_mode(){
      delete currentrun;
      delete nextrun;
      delete parser_strategy;        
    }

    //__attribute__((flatten)) // inlines all subsequent functions into this one, (potentially) increases speed at cost of larger code and compile time
    int run() override;
    void initialize() override;
};

#endif
