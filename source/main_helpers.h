/**
 * @file main_helpers.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This header contains functions that are potentially used throughout the whole program. TODO @TOM: Shall we leave it like this?
 * @version 0.1
 * @date 2023-03-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _MAIN_HELPERS_H_
#define _MAIN_HELPERS_H_

#include "state_merger.h"
#include "parameters.h"

#include "evaluation_factory.h"
#include "evaluate.h"

const bool debugging_enabled = false;

void print_current_automaton(state_merger* merger, const std::string& output_file, const std::string& append_string);
evaluation_function* get_evaluation();

#endif
