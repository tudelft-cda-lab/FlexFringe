/**
 * @file weight_comparator.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-01-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __WEIGHT_STATE_COMPARATOR__
#define __WEIGHT_STATE_COMPARATOR__

#include "weight_comparator.h"

#include <unordered_map>
#include <optional>

class weight_state_comparator_data: public weight_comparator_data {
    friend class weight_state_comparator;
protected:

    REGISTER_DEC_DATATYPE(weight_state_comparator_data);

    //float final_weight;
    //std::vector<float> symbol_weight_map;

    //double access_weight;

    vector<float> state;

public:
    weight_state_comparator_data() : weight_comparator_data::weight_comparator_data(){        
    }

    void initialize_state(const vector<float>& state){
        this->state = state;
    }

    //virtual void print_transition_label(iostream& output, int symbol) override;
    //virtual void print_state_label(iostream& output) override;

    //virtual void update(evaluation_data* right) override;
    //virtual void undo(evaluation_data* right) override;

    //virtual void read_json(json& node) override;
    //virtual void write_json(json& node) override;

    //virtual int predict_symbol(tail*) override;
    //virtual double predict_symbol_score(int t) override;

    //void set_weight(const int symbol, const float p);

    //void initialize_weights(const vector<int>& alphabet){
    //    symbol_weight_map.resize(alphabet.size());
    //    for(int i=0; i < alphabet.size(); ++i){
    //        symbol_weight_map[i] = 0;
    //    }
    //}

    //void initialize_access_weight(const double w){
    //    this->access_weight = w;
    //}

    //float get_final_weight() noexcept {
    //    return this->final_weight;
    //}

    //void set_final_weight(const float weight){
    //    this->final_weight = weight;
    //}

    //const std::vector<float>& get_weights() noexcept {
    //    return symbol_weight_map;
    //}

    //float get_weight(const int symbol) noexcept {
    //    return symbol_weight_map[symbol];
    //}

};

class weight_state_comparator: public weight_comparator {

protected:
    REGISTER_DEC_TYPE(weight_state_comparator);

    //float score;

public:

    virtual bool consistent(state_merger *merger, apta_node* left_node, apta_node* right_node, int depth) override;

    virtual double compute_score(state_merger*, apta_node* left_node, apta_node* right_node) override;
    virtual void reset(state_merger *merger) override;
    double compute_state_distance(apta_node* left_node, apta_node* right_node);
    
    static double get_distance(apta* aut, apta_node* left_node, apta_node* right_node);
};

#endif
