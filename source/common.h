/**
 * @file common.h
 * @author Sicco Verwer (s.e.verwer@tudelft.nl)
 * @brief Helper functions and macros to be used elsewhere.
 * @version 0.1
 * @date 2024-12-20
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _COMMON_H_
#define _COMMON_H_

#include "state_merger.h"
#include "parameters.h"
#include "evaluation_factory.h"
#include "evaluate.h"

#include <unordered_map>

const bool debugging_enabled = false;
extern char* gitversion;

#define DEBUG(x) do { \
  if (debugging_enabled) { std::cerr << x << std::endl; } \
} while (0)

evaluation_function* get_evaluation();

double update_score(double old_score, apta_node* next_node, tail* next_tail);
double compute_score(apta_node* next_node, tail* next_tail);
double compute_score(apta_node* old_node, apta_node* new_node);

apta_node* single_step(apta_node* n, tail* t, apta* a);


#endif 
