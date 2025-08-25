
#ifndef _DFASAT_H_
#define _DFASAT_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <list>
#include <sys/stat.h>
#include "state_merger.h"
#include "conflict_graph.h"

/**
 * @brief The merger context class. TODO: Legacy code that will be changed in the future. Get rid of merger context.
 *
 */
class dfasat {
public:
    dfasat(state_merger *m, int best_solution);

// apta/input state i has color j
    int **x;
    // color i has a transition with label a to color j
    int ***y;
    // color i has type t
    int **z;

    // color i has a transition with label a to sink j
    int ***sy;
    // state i is a sink or one of the parents of apta/input state i is a sink
    int *sp;

    // literals used for symmetry breaking
    int **yt;
    int **yp;
    int **ya;

    int literal_counter = 1;
    int clause_counter = 0;
    int alphabet_size = 0;

    bool computing_header = true;

    apta_graph* ag;
    state_merger* merger;

    state_set* red_states;
    state_set* non_red_states;
    state_set* sink_states;

    std::map<apta_node*, int> state_number;
    std::map<int, apta_node*> number_state;
    std::map<apta_node*, int> state_colour;

    std::stringstream sat_stream; // TODO: give some information on this object

    int dfa_size;
    int sinks_size;
    int num_states;
    int new_states;
    int new_init;
    int num_types;

    std::set<int> trueliterals;

    dfasat(state_merger* merger, std::string sat_program, int best_solution);

    void reset_literals(bool init);
    void create_literals();
    void delete_literals();
    int print_clause(bool v1, int l1, bool v2, int l2, bool v3, int l3, bool v4, int l4);
    int print_clause(bool v1, int l1, bool v2, int l2, bool v3, int l3);
    int print_clause(bool v1, int l1, bool v2, int l2);
    bool always_true(int number, bool flag);
    void print_lit(int number, bool flag);
    void print_clause_end();
    void fix_red_values();
    void fix_sink_values();
    int set_symmetry();
    int print_symmetry();
    void erase_red_conflict_colours();
    int print_colours();
    int print_conflicts();
    int print_accept();
    int print_transitions();
    int print_t_transitions();
    int print_p_transitions();
    int print_a_transitions();
    int print_forcing_transitions();
    int print_sink_transitions();
    int print_paths();
    int print_sink_paths();
    void print_dot_output(const char* dot_output);
    void print_aut_output(const char* aut_output);

    void compute_header();

    void read_solution(FILE *sat_file, int best_solution, state_merger*);

    void translate(FILE *sat_file);

    void perform_sat_merges(state_merger*);
};

void run_dfasat(state_merger* m, std::string sat_program, int best_solution);

#endif /* _DFASAT_H_ */
