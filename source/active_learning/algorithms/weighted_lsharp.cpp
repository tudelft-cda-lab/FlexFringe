/**
 * @file weighted_lsharp.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-11-29
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "weighted_lsharp.h"
#include "common_functions.h"

#include "inputdata.h"
#include "common.h"
#include "mem_store.h"
#include "parameters.h"
#include "state_merger.h"

#include "weight_comparator.h"

#include <list>
#include <stdexcept>

using namespace std;
using namespace active_learning_namespace;

/**
 * @brief Take a node and complete it wrt to the alphabet.
 *
 * This is the only function that creates new nodes (apart from the root node).
 *
 */
void weighted_lsharp_algorithm::extend_fringe(unique_ptr<state_merger>& merger, apta_node* n,
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
        seq.resize(1);
    
    auto* data = static_cast<weight_comparator_data*>(n->get_data());
    for (const int symbol : alphabet) {        
        seq[seq.size() - 1] = symbol;
        trace* new_trace = vector_to_trace(seq, id);
        id.add_trace_to_apta(new_trace, merger->get_aut(), false);

        query_weights(merger, n->get_child(symbol), id, alphabet, seq);

        // we do not create states that have zero incoming probability
        if(use_sinks && data->get_probability(symbol)== static_cast<double>(0)){
            auto* new_data = static_cast<weight_comparator_data*>(n->get_child(symbol)->get_data());
            new_data->set_sink();
        }
    }
    extended_nodes.insert(n);
}

/**
 * @brief Ask the oracle for the next distributoin and update the fields of node n with it.
 * seq_opt can be provided, minor optimization saving us runtime.
 */
void weighted_lsharp_algorithm::query_weights(unique_ptr<state_merger>& merger, apta_node* n, inputdata& id,
                                              const vector<int>& alphabet, optional<pref_suf_t> seq_opt) const {

    static unordered_set<apta_node*> completed_nodes;
    if (completed_nodes.contains(n))
        return;
    static const int SOS = START_SYMBOL;
    static const int EOS = END_SYMBOL;

    auto* data = static_cast<weight_comparator_data*>(n->get_data());
    data->initialize_weights(alphabet);

    pref_suf_t seq;
    if (seq_opt) {
        seq = pref_suf_t(seq_opt.value());
    } else {
        auto access_trace = n->get_access_trace();
        seq = access_trace->get_input_sequence(true, false);
    }

    const vector<double>& weights = oracle->ask_sul(seq, id).GET_DOUBLE_VEC();
    for (int i = 0; i < weights.size(); ++i) {
        if (SOS > -1 && i == SOS)
            continue;
        else if (i == EOS) {
            data->set_final_probability(weights[i]);
            continue;
        }
        int symbol = id.symbol_from_string(to_string(i));
        data->set_probability(symbol, weights[i]);
    }

    if (n == merger->get_aut()->get_root()) {
        data->init_access_probability(weights[EOS]);
    } else {
        auto node_it = merger->get_aut()->get_root();
        double w_product = 1;
        for (auto s : seq) {
            data = static_cast<weight_comparator_data*>(node_it->get_data());
            w_product *= data->get_probability(s);
            node_it = node_it->get_child(s);
        }
        assert(node_it == n);
        data = static_cast<weight_comparator_data*>(n->get_data());
        w_product *= weights[EOS];
        data->init_access_probability(w_product);
    }

    completed_nodes.insert(n);
}

/**
 * @brief Processing the counterexample recursively in the binary search strategy
 * as described by the paper.
 *
 * Operations done directly on the APTA.
 *
 * @param aut The merged APTA.
 * @param counterex The counterexample.
 */
void weighted_lsharp_algorithm::proc_counterex(inputdata& id, unique_ptr<apta>& hypothesis, const vector<int>& counterex,
                                               unique_ptr<state_merger>& merger, const refinement_list refs, const vector<int>& alphabet) const {
    // linear search to find fringe, then append new states
    cout << "proc counterex" << endl;
    reset_apta(merger.get(), refs);

    vector<int> substring;
    apta_node* n = hypothesis->get_root();
    for (auto s : counterex) {
        substring.push_back(s);
        trace* parse_trace = vector_to_trace(substring, id, 0); // TODO: inefficient like this, 0 is a dummy type that does not matter
        tail* t = substring.size() == 0 ? parse_trace->get_end() : parse_trace->get_end()->past_tail;
        apta_node* n_child = active_learning_namespace::get_child_node(n, t);
        
        if (n_child == nullptr) {
            query_weights(merger, n, id, alphabet, nullopt);
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
list<refinement*> weighted_lsharp_algorithm::find_complete_base(unique_ptr<state_merger>& merger,
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
            if(use_sinks && blue_node->is_sink()) continue; // we don't merge sinks

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
            find_closed_automaton(performed_refs, the_apta, merger, probabilistic_heuristic_interface::get_merge_distance_access_trace);
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
            find_closed_automaton(performed_refs, the_apta, merger, probabilistic_heuristic_interface::get_merge_distance_access_trace);
            return performed_refs;
        }

        //if (n_red_nodes >= MAX_RED_NODES && COUNTEREXAMPLE_STRATEGY){
        //    static const int MAX_ITER = 20;
        //    static int c_iter = 0;
        //    cout << "Continuing search using the counterexample strategy. Iteration " << c_iter << " out of " << MAX_ITER << endl;
        //    find_closed_automaton(performed_refs, the_apta, merger, weight_comparator::get_distance);
        //    c_iter++;
        //    if(c_iter==MAX_ITER){
        //        cout << "Max number of iterations reached. Printing automaton." << endl;
        //        print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
        //        exit(0);
        //    }
        //    return performed_refs;
        //}
    }
}

/**
 * @brief Main routine of this algorithm.
 * 
 * TODO: will be same as weighted L# and L# most likely
 *
 * @param id Inputdata.
 */
void weighted_lsharp_algorithm::run(inputdata& id) {
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
        query_weights(merger, root_node, id, alphabet, seq);
        extend_fringe(merger, root_node, the_apta, id, alphabet);
    }

    //{
    //    static int model_nr = 0;
    //    print_current_automaton(merger.get(), "model.", "root");
    //}

    while (ENSEMBLE_RUNS > 0 && n_runs <= ENSEMBLE_RUNS) {
        if (n_runs % 100 == 0)
            cout << "Iteration " << n_runs + 1 << endl;

        auto refs = find_complete_base(merger, the_apta, id, alphabet);
        cout << "Searching for counterexamples" << endl;

        // only merges performed, hence we can test our hypothesis
        while (true) {
            /* While loop to check the type. type is < 0 if sul cannot properly respond to query, e.g. when the string
            we ask cannot be parsed in automaton. We ignore those cases, as they lead to extra states in hypothesis.
            This puts a burden on the equivalence oracle to make sure no query is asked twice, else we end
            up in infinite loop.*/

            optional<pair<vector<int>, sul_response>> query_result = oracle->equivalence_query(merger.get());
            if (!query_result) {
                cout << "Found consistent automaton => Print." << endl;
                print_current_automaton(merger.get(), OUTPUT_FILE, ".final"); // printing the final model each time
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

            break;
        }

        // TODO: this one might not hold
        ++n_runs;
        if (ENSEMBLE_RUNS > 0 && n_runs == ENSEMBLE_RUNS) {
            cout << "Maximum of runs reached. Printing automaton." << endl;
            print_current_automaton(merger.get(), OUTPUT_FILE, ".final");
            return;
        }
    }
}
