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

protected:

    REGISTER_DEC_DATATYPE(log_alergia_data);
    std::unordered_map<int, double> symbol_probability_map;

public:
    virtual void print_transition_label(iostream& output, int symbol);
    virtual void print_state_label(iostream& output);

    virtual void update(evaluation_data* right);
    virtual void undo(evaluation_data* right);

    virtual void initialize();

    virtual void del_tail(tail *t);
    virtual void add_tail(tail *t);

    virtual void read_json(json& node);
    virtual void write_json(json& node);

    virtual int predict_symbol(tail*);
    virtual double predict_symbol_score(int t);
};

class log_alergia: public evaluation_function {

protected:
    REGISTER_DEC_TYPE(log_alergia);

public:

    virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right);

    virtual double compute_score(state_merger*, apta_node* left, apta_node* right);
    virtual void reset(state_merger *merger);

};

#endif
