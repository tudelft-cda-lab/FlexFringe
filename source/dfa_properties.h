//
// Created by sicco on 07/07/2022.
//

#ifndef FLEXFRINGE_DFA_PROPERTIES_H
#define FLEXFRINGE_DFA_PROPERTIES_H

#include "apta.h"
#include "input/inputdata.h"

bool is_counting_path(std::string str);
bool counting_path_occurs(apta_node* n1, apta_node* n2);
/** subtree is identical up to max_depth k, only using symbols
 * used in ktails implementations */
bool is_tree_identical(apta_node* l, apta_node* r, int max_depth);
/** path via sources is identical up to max_depth k, only using symbols
 * used in markovian and ngram implementations */
bool is_path_identical(apta_node* l, apta_node* r, int max_depth);
/** distances in nr of transitions to nodes in the apta and merged apta (approximation) */
int apta_distance(apta_node* l, apta_node* r, int bound);
int merged_apta_distance(apta_node* l, apta_node* r, int bound);
/** returns number of states targeting this state */
int num_distinct_sources(apta_node* node);

#endif //FLEXFRINGE_DFA_PROPERTIES_H
