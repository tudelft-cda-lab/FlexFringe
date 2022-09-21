
#ifndef _STATE_MERGER_H_
#define _STATE_MERGER_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <list>
#include <map>
#include <unordered_map>
#include <string>

class state_merger;

#include "evaluate.h"
#include "apta.h"
#include "refinement.h"
#include "mem_store.h"

using namespace std;

/**
 * @brief The state merger. Whereas the 
 * 
 */
class state_merger{
private:
    list<apta_node*> temporary_node_store;

    apta* aut;
    inputdata* dat;
    evaluation_function* eval;

    int num_merges = 0;

    map<int,apta_node*>* left_depth_map;
    map<int,apta_node*>* right_depth_map;

    /* recursive state merging routines */
    bool merge(apta_node* red, apta_node* blue);
    bool merge(apta_node* red, apta_node* blue, int depth, bool evaluate, bool perform, bool test);
    void merge_force(apta_node* red, apta_node* blue);
    bool merge_test(apta_node* red, apta_node* blue);
    void undo_merge(apta_node* red, apta_node* blue);

    /* recursive state splitting routines */
    bool split_single(apta_node* red, apta_node* blue, tail* t, int depth, bool evaluate, bool perform, bool test);
    void undo_split_single(apta_node* red, apta_node* blue);
    bool split(apta_node* red, apta_node* blue, int depth, bool evaluate, bool perform, bool test);
    void undo_split(apta_node* red, apta_node* blue);

    /* copy splits from red state before merging */
    void pre_split(apta_node* left, apta_node* right, int depth, bool evaluate, bool perform, bool test);
    void undo_pre_split(apta_node* red, apta_node* blue);

public:
    state_merger(inputdata*, evaluation_function*, apta*);
    ~state_merger();

    inline apta* get_aut(){
        return aut;
    }
    inline inputdata* get_dat(){
        return dat;
    }
    inline evaluation_function* get_eval(){
        return eval;
    }

    /* performing red-blue merges */
    void perform_merge(apta_node*, apta_node*); // merge function already above
    void undo_perform_merge(apta_node*, apta_node*);
    void perform_split(apta_node*, tail*, int);
    void undo_perform_split(apta_node*, tail*, int);

    /* creating new red states */
    void extend(apta_node* blue);
    void undo_extend(apta_node* blue);

    /* find refinements */
    refinement_set* get_possible_refinements();
    refinement* get_best_refinement();

    refinement* test_splits(apta_node* blue);
    refinement* test_merge(apta_node*,apta_node*);

    state_set* get_all_states() const;
    state_set* get_red_states() const;
    state_set* get_blue_states() const;
    state_set* get_candidate_states();
    state_set* get_sink_states();
    state_set* get_non_sink_states() const;

    int get_final_apta_size() const;
    int get_num_red_states() const;
    int get_num_red_transitions() const;

    void todot();
    void tojson();
    void print_dot(FILE*);
    void print_json(FILE*);

    int sink_type(apta_node* node);
    bool sink_consistent(apta_node* node, int type);
    int num_sink_types();

    string dot_output;
    string json_output;

    void tojsonsinks();

    refinement *test_split(apta_node *red, tail *t, int attr);

    apta_node *get_state_from_trace(trace *t) const;

    static trace *get_trace_from_state(apta_node *n);

    void undo_split_init(apta_node *red, tail *t, int attr);

    bool split_init(apta_node *red, tail *t, int attr, int depth, bool evaluate, bool perform, bool test);

    void print_dot(const string& file_name);

    void print_json(const string& file_name);

    int get_num_merges();

    void depth_check_init();

    void depth_check_fill(apta_node *node, map<int, apta_node *> *depth_map, int depth, bool use_symbol);

    bool depth_check_run(apta_node *left, apta_node *right, bool use_symbol);

    bool pre_consistent(apta_node *left, apta_node *right);

    double get_best_refinement_score();

    bool early_stop_merge(apta_node *left, apta_node *right, int depth, bool &val);

    void undo_split_single(apta_node *new_node, apta_node *old_node, tail *t);

    void renumber_states();
};

#endif /* _STATE_MERGER_H_ */
