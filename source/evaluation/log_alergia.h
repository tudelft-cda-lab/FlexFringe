/**
 * @file log_alergia.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-09-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __LOG_ALERGIA__
#define __LOG_ALERGIA__

#include "evaluate.h"

#include <unordered_map>

class log_alergia_data: public evaluation_data {
    friend class log_alergia;
protected:

    REGISTER_DEC_DATATYPE(log_alergia_data);
    std::unordered_map<int, double> symbol_probability_map;
    
    std::unordered_map<int, double> final_symbol_probability_map;
    double final_prob;

public:
    virtual void print_transition_label(iostream& output, int symbol) override;
    virtual void print_state_label(iostream& output) override;

    virtual void update(evaluation_data* right) override;
    virtual void undo(evaluation_data* right) override;

    virtual void read_json(json& node) override;
    virtual void write_json(json& node) override;

    virtual int predict_symbol(tail*) override;
    virtual double predict_symbol_score(int t) override;

    double get_probability(const int symbol);
    void insert_probability(const int symbol, const double p);

    double get_final_probability(const int symbol);
    void insert_final_probability(const int symbol, const double p);

    void update_final_prob(const double p) noexcept {
        this->final_prob += p;
    }

    std::unordered_map<int, double>& get_final_distribution(){
        return final_symbol_probability_map;
    }
};

class log_alergia: public evaluation_function {

protected:
    REGISTER_DEC_TYPE(log_alergia);

    double mu;
    double js_divergence;

public:

    virtual bool consistent(state_merger *merger, apta_node* left_node, apta_node* right_node) override;


    virtual double compute_score(state_merger*, apta_node* left_node, apta_node* right_node) override;
    virtual void reset(state_merger *merger) override;
    virtual void initialize_before_adding_traces() override;

    static void normalize_final_probs(apta_node* node);
    static double get_js_term(const double px, const double qx);
    static double get_js_divergence(unordered_map<int, double>& left_distribution, std::unordered_map<int, double>& right_distribution);
};

#endif
