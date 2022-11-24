#ifndef __ALERGIA__
#define __ALERGIA__

#include "evaluate.h"
#include "count_types.h"

/* The data contained in every node of the prefix tree or DFA */
class alergia_data: public count_data {

protected:

    REGISTER_DEC_DATATYPE(alergia_data);

public:
    num_map symbol_counts;

    inline int count(int symbol){
        auto hit = symbol_counts.find(symbol);
        if(hit == symbol_counts.end()) return 0;
        return hit->second;
    }

    inline num_map& get_symbol_counts(){
        return symbol_counts;
    }

    inline num_map::iterator counts_begin(){
        return symbol_counts.begin();
    }

    inline num_map::iterator counts_end(){
        return symbol_counts.end();
    }

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
    virtual tail* sample_tail();

    virtual double align_score(tail *t);

    void get_symbol_divider(double &divider, double &count);

    void get_type_divider(double &divider, double &count);

};

class alergia: public count_driven {

protected:
    REGISTER_DEC_TYPE(alergia);

    double num_tests;
    double sum_diffs;

    static void update_divider(double left_count, double right_count, double &left_divider, double &right_divider);
    static void update_divider_pool(double left_pool, double right_pool, double &left_divider, double &right_divider);
    static void update_left_pool(double left_count, double right_count, double &left_pool, double &right_pool);
    static void update_right_pool(double left_count, double right_count, double &left_pool, double &right_pool);

    virtual bool test_and_update(double right_count, double left_count, double right_total, double left_total);

public:

    virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right);

    virtual double compute_score(state_merger*, apta_node* left, apta_node* right);
    virtual void reset(state_merger *merger);

    static double alergia_check(double right_count, double left_count, double right_total, double left_total);

    bool pool_and_compute_tests(num_map& left_map, int left_total, int left_final, num_map& right_map, int right_total, int right_final);

    bool prob_consistency(alergia_data *l, alergia_data *r);
};

#endif
