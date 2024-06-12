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

#include <unordered_map>
#include <optional>

class weight_comparator_data: public evaluation_data {
    friend class weight_comparator;
protected:

    REGISTER_DEC_DATATYPE(weight_comparator_data);

    float final_weight;
    std::vector<float> symbol_weight_map;

    double access_weight;
    bool is_sink;

public:
    weight_comparator_data() : evaluation_data::evaluation_data(){        
        final_weight = 0;
        access_weight = 0;
        is_sink = false;
    }
    virtual void print_transition_label(iostream& output, int symbol) override;
    virtual void print_state_label(iostream& output) override;

    virtual void update(evaluation_data* right) override;
    virtual void undo(evaluation_data* right) override;

    virtual void read_json(json& node) override;
    virtual void write_json(json& node) override;

    virtual int predict_symbol(tail*) override;
    virtual double predict_symbol_score(int t) override;

    const float get_weight(const int symbol) const;
    void set_weight(const int symbol, const float p);

    void initialize_weights(const vector<int>& alphabet){
        symbol_weight_map.resize(alphabet.size());
        for(int i=0; i < alphabet.size(); ++i){
            symbol_weight_map[i] = 0;
        }
    }

    void initialize_access_weight(const double w){
        this->access_weight = w;
    }

    float get_final_weight() noexcept {
        return this->final_weight;
    }

    void set_final_weight(const float weight){
        this->final_weight = weight;
    }

    const std::vector<float>& get_weights() noexcept {
        return symbol_weight_map;
    }

    float get_weight(const int symbol) noexcept {
        return symbol_weight_map[symbol];
    }

    double get_access_probability() const noexcept;

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

    float score;

public:

    virtual bool consistent(state_merger *merger, apta_node* left_node, apta_node* right_node, int depth) override;

    virtual double compute_score(state_merger*, apta_node* left_node, apta_node* right_node) override;
    virtual void reset(state_merger *merger) override;

    static double get_distance(apta* aut, apta_node* left_node, apta_node* right_node);
};

#endif
