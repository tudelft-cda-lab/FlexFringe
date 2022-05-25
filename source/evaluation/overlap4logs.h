#ifndef __OVERLAP4LOGS__
#define __OVERLAP4LOGS__

#include "overlap.h"

typedef map<long, long> long_map;
typedef map<int, map<long, long>> num_long_map;

/* The data contained in every node of the prefix tree or DFA */
class overlap4logs_data: public overlap_data {

public:
    num_map num_type;
    num_long_map num_delays;

    inline int types(int i){
        num_map::iterator it = num_type.find(i);
        if(it == num_type.end()) return 0;
        return it->second;
    }

    overlap4logs_data();

    set<string> trace_ids;
    virtual void store_id(string id);

    virtual void add_tail(tail* t);
    virtual double delay_mean(int symbol);
    virtual double delay_std(int symbol);
    virtual void update(evaluation_data* right);
    virtual void undo(evaluation_data* right);

    virtual int sink_type();
    virtual bool sink_consistent(int type);
    virtual int num_sink_types();

    virtual void print_state_label(iostream& output);
    virtual void print_state_style(iostream& output);
    virtual void print_transition_label(iostream& output, int symbol);
    virtual void print_transition_style(iostream& output, set<int> symbols);
    virtual int find_end_type(apta_node* node);

protected:
    REGISTER_DEC_DATATYPE(overlap4logs_data);
};

class overlap4logs: public overlap_driven {

protected:
    REGISTER_DEC_TYPE(overlap4logs);

public:
    virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right);
    virtual void update_score(state_merger *merger, apta_node* left, apta_node* right);
    virtual double compute_score(state_merger*, apta_node* left, apta_node* right);
    virtual int print_labels(iostream& output, apta* aut, overlap4logs_data* data, int symbol);
    //virtual void print_dot(iostream& output, state_merger* merger);
};

#endif
