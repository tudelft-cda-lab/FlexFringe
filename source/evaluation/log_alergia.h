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

public:
    virtual void print_transition_label(iostream& output, int symbol) override;
    //virtual void print_state_label(iostream& output) override;

    virtual void update(evaluation_data* right) override;
    virtual void undo(evaluation_data* right) override;

    //virtual void initialize() override;

    //virtual void del_tail(tail *t) override;
    //virtual void add_tail(tail *t) override;

    virtual void read_json(json& node) override;
    virtual void write_json(json& node) override;

    virtual int predict_symbol(tail*) override;
    virtual double predict_symbol_score(int t) override;

    double get_probability(const int symbol);
    void insert_probability(const int symbol, const double p);
};

class log_alergia: public evaluation_function {

protected:
    REGISTER_DEC_TYPE(log_alergia);

    double log_mu; 
    double log_sqr_n;
    double alpha;
    double right;

    double sum_diffs;

    
public:

    virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right) override;

    virtual double compute_score(state_merger*, apta_node* left, apta_node* right) override;
    virtual void reset(state_merger *merger) override;
    virtual void initialize_before_adding_traces() override;

    double get_log_mu(){
        return log_mu;
    }
};

#endif
