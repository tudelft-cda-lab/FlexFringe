/**
 * @file predict_mode.h
 * @author Sicco Verwer (s.e.verwer@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-18
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _PREDICT_MODE_H
#define _PREDICT_MODE_H

#include "running_mode_base.h"
#include "state_merger.h"
#include "input/inputdata.h"

#include <unordered_map>
#include <string>
#include <fstream>

class predict_mode : public running_mode_base {
  private:
    struct tail_state_compare;

    std::list<int> state_sequence;
    std::list<double> score_sequence;
    std::list<bool> align_sequence;
    apta_node* ending_state = nullptr;
    tail* ending_tail = nullptr;

    void align(state_merger* m, tail* t, bool always_follow, double lower_bound);
    double prob_single_parallel(tail* p, tail* t, apta_node* n, double prod_prob, bool flag);

    double compute_skip_penalty(apta_node* node);
    double compute_jump_penalty(apta_node* old_node, apta_node* new_node);

    void predict_csv(state_merger* m, std::istream& input, std::ostream& output);
    void predict(state_merger* m, inputdata& idat, std::ostream& output, parser* input_parser);
    void predict_trace_update_sequences(state_merger* m, tail* t);
    void predict_streaming(state_merger* m, parser& parser, reader_strategy& strategy, std::ofstream& output);
    
    void predict_trace(state_merger* m, std::ostream& output, trace* tr);
    [[maybe_unused]] double predict_trace(state_merger* m, trace* tr);
    void predict_header(std::ostream& output);

    void store_visited_states(apta_node* n, tail* t, state_set* states);
    double add_visits(state_merger* m, trace* tr);
    std::pair<int,int> visited_node_sizes(apta_node* n, tail* t);


    template <typename T>
    void write_list(std::list<T>& list_to_write, std::ostream& output);
  
  public:
    int run() override;
    void initialize() override;

    std::unordered_map<std::string, std::string> get_prediction_mapping(state_merger* m, trace* tr);
};

template <typename T>
void predict_mode::write_list(std::list<T>& list_to_write, std::ostream& output){
    if(list_to_write.empty()){
        output << "[]";
        return;
    }

    output << "[";
    bool first = true;
    for(auto val : list_to_write) {
        if (!first) { output << ","; }
        else first = false;
        output << val;
    }
    output << "]";
}

#endif //_PREDICT_MODE_H
