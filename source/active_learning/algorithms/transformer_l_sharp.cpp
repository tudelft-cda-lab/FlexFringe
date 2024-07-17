/**
 * @file transformer_l_sharp.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-06-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "transformer_l_sharp.h"
#include "base_teacher.h"
#include "common_functions.h"
#include "input_file_oracle.h"
#include "input_file_sul.h"

#include "greedy.h"
#include "inputdata.h"
#include "main_helpers.h"
#include "mem_store.h"
#include "parameters.h"
#include "state_merger.h"

#include "edsm.h"

#include <list>

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Take a node and complete it wrt to the alphabet.
 *
 * This is the only function that creates new nodes (apart from the root node).
 *
 */
void transformer_lsharp_algorithm::extend_fringe(unique_ptr<state_merger>& merger, apta_node* n,
                                                                   unique_ptr<apta>& the_apta, inputdata& id,
                                                                   const vector<int>& alphabet) const {
    static unordered_set<apta_node*> extended_nodes;
    if (extended_nodes.contains(n))
        return;

    auto access_trace = n->get_access_trace();
    pref_suf_t seq;
    if (n->get_number() != -1 && n->get_number() != 0)
        seq = access_trace->get_input_sequence(true, true);
    else
        seq.resize(1); // this is the root node

    auto* data = static_cast<membership_state_comparator_data*>(n->get_data());
    for (const int symbol : alphabet) {        
        seq[seq.size() - 1] = symbol;
    
        const pair< int, vector< vector<float> > > answer = teacher->get_membership_state_pair(seq, id);
        auto type = answer.first;
        
        trace* new_trace = vector_to_trace(seq, id, type);
        id.add_trace_to_apta(new_trace, merger->get_aut(), false);

        update_hidden_states(merger, n->get_child(symbol), id, alphabet, answer.second, seq);

        // create sink states that we do not consider for merges
        //if(use_sinks && data->get_weight(symbol)==float(0.0)){
            //auto* new_data = static_cast<membership_state_comparator_data*>(n->get_child(symbol)->get_data());
            //new_data->set_sink();
        //}
    }

    extended_nodes.insert(n);
}

/**
 * @brief Adds statistics to a node, returns the traces that were queried along the way.
 *
 * @param n The node to upate.
 * @param hidden_states Shape=(len_sequence, hidden_state_dim)
 * @return optional< vector<trace*> > A vector with all the new traces we added to the apta, or nullopt if we already
 * processed this node. Saves runtime.
 */
void transformer_lsharp_algorithm::update_hidden_states(unique_ptr<state_merger>& merger, apta_node* n, inputdata& id, const vector<int>& alphabet, 
                                              const optional< vector< vector<float > > >& hidden_states_opt, optional<pref_suf_t> seq_opt) const {

    static unordered_set<apta_node*> completed_nodes;
    if (completed_nodes.contains(n))
        return;
    
    pref_suf_t seq;
    trace* access_trace = n->get_access_trace();
    if (seq_opt)
        seq = pref_suf_t(seq_opt.value());
    else
        seq = access_trace->get_input_sequence(true, false);

    vector< vector<float> > hidden_states;
    if(hidden_states_opt)
        hidden_states = hidden_states_opt.value();
    else{
        const pair< int, vector< vector<float> > > answer = teacher->get_membership_state_pair(seq, id);
        hidden_states = answer.second;
    }

    if (n == merger->get_aut()->get_root()) {
        auto* data = static_cast<membership_state_comparator_data*>(n->get_data());
        data->update_sums(hidden_states[0]);
    } else {
        tail* t = access_trace->get_head();
        int idx = 0;
        auto n_it = merger->get_aut()->get_root();
        while(true){
            auto* data = static_cast<membership_state_comparator_data*>(n_it->get_data());
            data->update_sums(hidden_states[idx]);
            if(t==nullptr) // n==n_it now
                break;

            ++idx;
            auto s = t->get_symbol();
            n_it = n_it->get_child(s);
            t = t->future(); // next()?
        }
    }

    completed_nodes.insert(n);
}

/**
 * @brief Does what you think it does.
 *
 * @param merger
 * @param the_apta
 * @return list<refinement*>
 */
list<refinement*> transformer_lsharp_algorithm::find_complete_base(unique_ptr<state_merger>& merger,
                                                                unique_ptr<apta>& the_apta, inputdata& id,
                                                                const vector<int>& alphabet) {
    static const bool COUNTEREXAMPLE_STRATEGY = false;

    int n_red_nodes = 1; // for the root node
    static const int MAX_RED_NODES = 2500;
    bool termination_reached = false;
    int n_iter = -1;

    list<refinement*> performed_refs;
    while (true) { // cancel when either not red node identified or max number of nodes is reached

        unordered_set<apta_node*> blue_nodes;
        unordered_set<apta_node*> red_nodes;

        cout << ++n_iter << " iterations for this round. Red nodes: " << n_red_nodes << endl;
        
        for (blue_state_iterator b_it = blue_state_iterator(the_apta->get_root()); *b_it != nullptr; ++b_it) {
            auto blue_node = *b_it;
            blue_nodes.insert(blue_node);
        }

        for (red_state_iterator r_it = red_state_iterator(the_apta->get_root()); *r_it != nullptr; ++r_it) {
            const auto red_node = *r_it;
            red_nodes.insert(red_node);
        }

        bool identified_red_node = false;
        for (auto blue_node : blue_nodes) {
            //if(use_sinks && blue_node->is_sink()) continue; // we don't merge sinks

            refinement_set possible_merges;
            for(auto red_node: red_nodes){
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

                // we need to add to an unmerged apta
                reset_apta(merger.get(), performed_refs);
                extend_fringe(merger, blue_node, the_apta, id, alphabet);
                do_operations(merger.get(), performed_refs);

                if(n_red_nodes == MAX_RED_NODES){
                    termination_reached = true;
                    break;
                }
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
            find_closed_automaton(performed_refs, the_apta, merger, evidence_driven::get_score); // TODO: does this one make sense here?
            print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
            cout << "Printed. Terminating" << endl;
            exit(0);
        }

        //{
        //    static int model_nr = 0;
        //    cout << "printing model " << model_nr  << endl;
        //    print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".after_refs");
        //}

        if(!identified_red_node){
            //static int n_h = 0;
            //++n_h;
            //if(n_h==50){
            //    cout << "Max number of hypotheses reached. Printing the automaton with " << n_red_nodes << " states." << endl;
            //    find_closed_automaton(performed_refs, the_apta, merger, weight_comparator::get_distance);
            //    print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
            //    cout << "Printed. Terminating" << endl;
            //    exit(0);
            //}
            cout << "Complete basis found. Forwarding hypothesis" << endl;
            find_closed_automaton(performed_refs, the_apta, merger, evidence_driven::get_score);
            return performed_refs;
        }
    }
}

/**
 * @brief Main routine of this algorithm.
 * 
 * TODO: will be same as weighted L# and L# most likely
 *
 * @param id Inputdata.
 */
void transformer_lsharp_algorithm::run(inputdata& id) {
    MERGE_WHEN_TESTING = true;
    cout << "testmerge option set to true, because algorithm relies on it" << endl;

    int n_runs = 1;

    auto eval = unique_ptr<evaluation_function>(get_evaluation());
    eval->initialize_before_adding_traces();

    auto the_apta = unique_ptr<apta>(new apta());
    auto merger = unique_ptr<state_merger>(new state_merger(&id, eval.get(), the_apta.get()));
    this->oracle->initialize(merger.get());

    const vector<int> alphabet = id.get_alphabet();
    cout << "Alphabet: ";
    active_learning_namespace::print_sequence<vector<int>::const_iterator>(alphabet.cbegin(), alphabet.cend());

    {
        // init the root node
        auto root_node = the_apta->get_root();
        pref_suf_t seq;
        const pair< int, vector< vector<float> > > answer = teacher->get_membership_state_pair(seq, id);
        auto type = answer.first;
        
        trace* new_trace = vector_to_trace(seq, id, type);
        id.add_trace_to_apta(new_trace, merger->get_aut(), false);

        update_hidden_states(merger, root_node, id, alphabet, answer.second, seq);
        cout << "Lenght of 1: " << answer.second.size() << endl;
        
        extend_fringe(merger, root_node, the_apta, id, alphabet);
    }

    /* {
        print_current_automaton(merger.get(), "model.", "root");
    } */

    while (ENSEMBLE_RUNS > 0 && n_runs <= ENSEMBLE_RUNS) {
        if (n_runs % 100 == 0)
            cout << "Iteration " << n_runs + 1 << endl;

        auto refs = find_complete_base(merger, the_apta, id, alphabet);
        cout << "Searching for counterexamples" << endl;

        {
            static int model_nr = 0;
            print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".before_cex");
        }

        // only merges performed, hence we can test our hypothesis
        while (true) {
            /* While loop to check the type. type is < 0 if sul cannot properly respond to query, e.g. when the string
            we ask cannot be parsed in automaton. We ignore those cases, as they lead to extra states in hypothesis.
            This puts a burden on the equivalence oracle to make sure no query is asked twice, else we end
            up in infinite loop.*/

            optional<pair<vector<int>, int>> query_result = oracle->equivalence_query(merger.get(), teacher);
            if (!query_result) {
                cout << "Found consistent automaton => Print." << endl;
                print_current_automaton(merger.get(), OUTPUT_FILE, ".final"); // printing the final model each time
                return;
            }

            const int type = query_result.value().second;
            if (type < 0)
                continue;

            const vector<int>& cex = query_result.value().first;
            cout << "Counterexample of length " << cex.size() << " found: ";
            for(auto s: cex)
                cout << id.get_symbol(s) << " ";
            cout << endl;

            proc_counterex(teacher, id, the_apta, cex, merger, refs, alphabet);

            {
                static int model_nr = 0;
                print_current_automaton(merger.get(), "model.", to_string(++model_nr) + ".after_cex");
            }
            break;
        }
    }
}
