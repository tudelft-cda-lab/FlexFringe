/**
 * @file weight_comparator.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-09-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __WEIGHT_COMPARATOR__
#define __WEIGHT_COMPARATOR__

#include "evaluate.h"
#include "probabilistic_heuristic_interface.h"

#include <unordered_map>
#include <optional>

class weight_comparator_data: public probabilistic_heuristic_interface {
    friend class weight_comparator;

protected:
    REGISTER_DEC_DATATYPE(weight_comparator_data);
    bool is_sink;

public:
    weight_comparator_data() : evaluation_data::evaluation_data(), probabilistic_heuristic_interface(){        
        final_prob = 0;
        access_trace_prob = 0;
        is_sink = false;
    }

    void print_transition_label(std::iostream& output, int symbol) override;
    void print_state_label(std::iostream& output) override;

    void update(evaluation_data* right) override;
    void undo(evaluation_data* right) override;

    void read_json(json& node) override;
    void write_json(json& node) override;

    int predict_symbol(tail*) override;
    double predict_symbol_score(int t) override;

    void initialize_weights(const std::vector<int>& alphabet){
        symbol_probability_map.resize(alphabet.size());
        for(int i=0; i < alphabet.size(); ++i){
            symbol_probability_map[i] = 0;
        }
    }

    int sink_type() noexcept override {
        return is_sink ? 1 : -1;
    }

    void set_sink() noexcept {
        is_sink = true;
    }

};

class weight_comparator: public evaluation_function {

protected:
    REGISTER_DEC_TYPE(weight_comparator);

    double score;

public:

    bool consistent(state_merger *merger, apta_node* left_node, apta_node* right_node, int depth) override;

    double compute_score(state_merger*, apta_node* left_node, apta_node* right_node) override;
    void reset(state_merger *merger) override;

    static double get_distance(apta* aut, apta_node* left_node, apta_node* right_node);
};

#endif
