/**
 * @file lsharp.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The (strategic) L#-algorithm, as described by Vandraager et al. (2022): "A New Approach for Active Automata
 * Learning Based on Apartness"
 * @version 0.1
 * @date 2023-03-11
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "lsharp.h"
#include "common_functions.h"

#include "inputdata.h"
#include "output_manager.h"
#include "common.h"
#include "mem_store.h"
#include "parameters.h"
#include "state_merger.h"

#include "edsm.h"

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief For inheritance reasons. Saves us a second proc_counterexample implementation in derived classes.
 */
void lsharp_algorithm::update_state(std::unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
                                    const std::vector<int>& alphabet) const {
    return;
}

/**
 * @brief Take a node and complete it wrt to the alphabet.
 *
 * This is the only function that creates new nodes (apart from the root node).
 *
 */
void lsharp_algorithm::extend_fringe(unique_ptr<state_merger>& merger, apta_node* n,
                                                                   unique_ptr<apta>& the_apta, inputdata& id,
                                                                   const vector<int>& alphabet) const {
    static unordered_set<apta_node*> extended_nodes; // TODO: do we need this data structure here?
    if (extended_nodes.contains(n))
        return;

    auto access_trace = n->get_access_trace();
    pref_suf_t seq;
    if (n->get_number() != -1 && n->get_number() != 0)
        seq = access_trace->get_input_sequence(true, true);
    else
        seq.resize(1); // this is the root node

    for (const int symbol : alphabet) {
        seq[seq.size() - 1] = symbol;

        const int answer = oracle->ask_sul(seq, id).GET_INT();
        if(answer==-1)
            continue;

        trace* new_trace = vector_to_trace(seq, id, answer);
        id.add_trace_to_apta(new_trace, merger->get_aut(), false);
        assert(n->get_child(symbol) != nullptr);
    }

    extended_nodes.insert(n);
}

/**
 * @brief Processing the counterexample.
 * 
 * TODO: Make this one better.
 *
 * Operations done directly on the APTA.
 *
 * @param aut The merged APTA.
 * @param counterex The counterexample.
 */
void lsharp_algorithm::proc_counterex(inputdata& id,
                                      unique_ptr<apta>& hypothesis, const vector<int>& counterex,
                                      unique_ptr<state_merger>& merger, const refinement_list refs,
                                      const vector<int>& alphabet) const {
    // in this block we do a linear search for the fringe of the prefix tree. Once we found it, we ask membership
    // queries for each substring of the counterexample (for each new state that we create), and this way add the whole
    // counterexample to the prefix tree
    reset_apta(merger.get(), refs);
    vector<int> substring;
    apta_node* n = hypothesis->get_root();
    for (auto s : counterex) {

        substring.push_back(s);
        trace* parse_trace = vector_to_trace(substring, id, 0); // TODO: inefficient like this, 0 is a dummy type that does not matter
        tail* t = substring.size() == 0 ? parse_trace->get_end() : parse_trace->get_end()->past_tail;
        apta_node* n_child = active_learning_namespace::get_child_node(n, t);

        if (n_child == nullptr) {
            extend_fringe(merger, n, hypothesis, id, alphabet);
            n_child = active_learning_namespace::get_child_node(n, t);
        }

        n = n_child;
        mem_store::delete_trace(parse_trace);
    }
}



/**
 * @brief Does what you think it does.
 * 
 * TODO: will be same as weighted L# and L# most likely
 *
 * @param merger
 * @param the_apta
 * @return list<refinement*>
 */
list<refinement*> lsharp_algorithm::find_complete_base(unique_ptr<state_merger>& merger,
                                                                unique_ptr<apta>& the_apta, inputdata& id,
                                                                const vector<int>& alphabet) {
    static const bool COUNTEREXAMPLE_STRATEGY = false;
    static const bool merge_root = MERGE_ROOT;

    int n_red_nodes = 1; // for the root node
    bool termination_reached = false;
    int n_iter = -1;

    list<refinement*> performed_refs;
    while (true) { // cancel when either not red node identified or max number of nodes is reached

        unordered_set<apta_node*> blue_nodes;

        cout << ++n_iter << " iterations for this round. Red nodes: " << n_red_nodes << endl;
        
        for (blue_state_iterator b_it = blue_state_iterator(the_apta->get_root()); *b_it != nullptr; ++b_it) {
            auto blue_node = *b_it;
            blue_nodes.insert(blue_node);
        }

        bool identified_red_node = false;
        for (auto blue_node : blue_nodes) {
            refinement_set possible_merges;

            for(red_state_iterator r_it = red_state_iterator(the_apta->get_root()); *r_it != nullptr; ++r_it){
                const auto red_node = *r_it;
                if(!merge_root && red_node == the_apta->get_root())
                    continue;

                refinement* ref = merger->test_merge(red_node, blue_node);
                if (ref != nullptr)
                    possible_merges.insert(ref);
            }

            if (possible_merges.size() == 0) {
                identified_red_node = true;
                refinement* ref = mem_store::create_extend_refinement(merger.get(), blue_node);
                ref->doref(merger.get());
                performed_refs.push_back(ref);

                ++n_red_nodes;

                // we have to have the unmerged tree when creating new states. Else errors might occur
                reset_apta(merger.get(), performed_refs);
                extend_fringe(merger, blue_node, the_apta, id, alphabet);
                do_operations(merger.get(), performed_refs);

            /* } else if (IDENTIFY_STATE_COMPLETELY && possible_merges.size() > 1) { */

            } else {
                // get the best merge from the heap
                refinement* best_merge = *(possible_merges.begin());
                for (auto it : possible_merges) {
                    if (it != best_merge)
                        it->erase();
                }
                best_merge->doref(merger.get());
                performed_refs.push_back(best_merge);
            }
        }

        if(termination_reached){
            cout << "Max number of states reached and counterexample strategy disabled. Printing the automaton with " << n_red_nodes << " states." << endl;
            find_closed_automaton(performed_refs, the_apta, merger, evidence_driven::get_score);
            output_manager::print_final_automaton(merger.get(), ".final");
            cout << "Printed. Terminating" << endl;
            exit(0);
        }

        //{
        //    static int model_nr = 0;
        //    cout << "printing model " << model_nr  << endl;
        //    output_manager::print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".after_refs");
        //}

        if(!identified_red_node){
            cout << "Complete basis found. Forwarding hypothesis" << endl;
            find_closed_automaton(performed_refs, the_apta, merger, evidence_driven::get_score);
            return performed_refs;
        }
    }
}

void lsharp_algorithm::run(inputdata& id) {
    auto eval = unique_ptr<evaluation_function>(get_evaluation());
    eval->initialize_before_adding_traces();

    auto the_apta = unique_ptr<apta>(new apta());
    auto merger = unique_ptr<state_merger>(new state_merger(&id, eval.get(), the_apta.get()));
    this->oracle->initialize(merger.get());
    set_types();

    const vector<int> alphabet = id.get_alphabet();
    cout << "Alphabet: ";
    active_learning_namespace::print_sequence<vector<int>::const_iterator>(alphabet.cbegin(), alphabet.cend());

    {
        // init the root node, s.t. we have blue states to iterate over
        if(MERGE_ROOT){
            pref_suf_t seq;
            const int answer = oracle->ask_sul(seq, id).GET_INT();
            trace* new_trace = vector_to_trace(seq, id, answer);
            id.add_trace_to_apta(new_trace, merger->get_aut(), false);
        }

        auto root_node = the_apta->get_root();
        extend_fringe(merger, root_node, the_apta, id, alphabet);
    }

    while (true) {
        auto refs = find_complete_base(merger, the_apta, id, alphabet);
        cout << "Searching for counterexamples" << endl;

        static const int MAX_N_NODES = AL_MAX_N_STATES;
        const auto n_nodes = count_nodes(the_apta.get());
        if(MAX_N_NODES > -1 && n_nodes >= MAX_N_NODES){
            cout << "Reached maximum number of states. Printing." << endl;
            output_manager::print_final_automaton(merger.get(), ".final");
            return;
        }

        /* {
            static int model_nr = 0;
            output_manager::print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".before_cex");
        } */

        // only merges performed, hence we can test our hypothesis
        while (true) {
            /* While loop to check the type. type is < 0 if sul cannot properly respond to query, e.g. when the string
            we ask cannot be parsed in automaton. We ignore those cases, as they lead to extra states in hypothesis.
            This puts a burden on the equivalence oracle to make sure no query is asked twice, else we end
            up in infinite loop.*/

            optional<pair<vector<int>, sul_response>> query_result = oracle->equivalence_query(merger.get());
            if (!query_result) {
                cout << "Found consistent automaton => Print." << endl;
                output_manager::print_final_automaton(merger.get(), ".final"); // printing the final model each time
                return;
            }

            const int type = query_result.value().second.get_int();
            if (type < 0)
                continue;

            const vector<int>& cex = query_result.value().first;
            cout << "Counterexample of length " << cex.size() << " found: ";
            for(auto s: cex)
                cout << id.get_symbol(s) << " ";
            cout << endl;        
            
            proc_counterex(id, the_apta, cex, merger, refs, alphabet);

            /* {
                static int model_nr = 0;
                output_manager::print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".after_cex");
            } */

            break;
        }
    }
}
