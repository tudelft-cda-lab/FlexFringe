//
// Created by sicco on 23/04/2021.
//
#ifndef FLEXFRINGE_PREDICT_H
#define FLEXFRINGE_PREDICT_H

#include "state_merger.h"
#include "input/inputdata.h"

apta_node* single_step(apta_node* n, tail* t, apta* a);
double compute_score(apta_node* next_node, tail* next_tail);
[[maybe_unused]] double predict_trace(state_merger* m, trace* tr);


[[maybe_unused]] void predict(state_merger* m, inputdata& idat, std::ofstream& output);
void predict_streaming(state_merger* m, parser& parser, reader_strategy& strategy, std::ofstream& output);

#endif //FLEXFRINGE_PREDICT_H
