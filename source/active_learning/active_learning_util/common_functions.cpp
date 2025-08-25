/**
 * @file common_functions.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-02-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "common_functions.h"
#include "definitions.h"
#include "inputdata.h"
#include "inputdatalocator.h"
#include "parameters.h"

#include "string_probability_estimator.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <limits>

using namespace std;

// const bool PRINT_ALL_MODELS = false;

apta_node* active_learning_namespace::get_child_node(apta_node* n, tail* t) {
    apta_node* child = n->child(t);
    if (child == 0) {
        return nullptr;
    }
    return child->find();
}

apta_node* active_learning_namespace::get_child_node(apta_node* n, int symbol) {
    apta_node* child = n->child(symbol);
    return child == nullptr ? nullptr : child->find();
}

/**
 * @brief We use this function e.g. to determine termination of an algorithm.
 */
int active_learning_namespace::count_nodes(apta* aut){
    int res = 0;
    for(auto it = red_state_iterator(aut->get_root()); *it!=nullptr; ++it)
        res++;
        
    return res;
}

/**
 * @brief There are two versions of this function. In this version we look at if the tree is
 * possibly parsable by the traces.
 *
 * The problem with this one is that in algorithms where we expect positive and negative traces,
 * then the resulting DFA will be the root node with lots of self-loops only. Reason: This way we
 * won't get a counterexample, all traces will be accepted.
 *
 * @param tr The trace.
 * @param aut The apta.
 * @return true Accepts trace.
 * @return false Does not accept trace.
 */
bool active_learning_namespace::aut_accepts_trace(trace* tr, apta* aut) {
    apta_node* n = aut->get_root();
    tail* t = tr->get_head();
    for (int j = 0; j < t->get_length(); j++) {
        n = active_learning_namespace::get_child_node(n, t);
        if (n == nullptr)
            return false; // TODO: does this one make sense?

        t = t->future();
    }
    return true;
}

/**
 * @brief Predicts type from trace. Code is a subset of the predict_mode.cpp functions, hence duplicate code. TODO: clean
 * that up
 *
 * @param tr The trace.
 * @return const int The predicted type.
 */
const int active_learning_namespace::predict_type_from_trace(trace* tr, apta* aut, inputdata& id) {
    apta_node* n = aut->get_root();
    tail* t = tr->get_head(); // TODO: get
    for (int j = 0; j < t->get_length(); j++) {
        n = get_child_node(n, t);
        if (n == nullptr) {
            return -1; // Invariant: inputdata will never map to negative values.
        }
        t = t->future();
    }

    apta_node* ending_state = n;
    tail* ending_tail = t;
    if (ending_tail->is_final())
        ending_tail = ending_tail->past();

    return ending_state->get_data()->predict_type(ending_tail);
}

apta_node* active_learning_namespace::get_last_node(trace* tr, apta* aut, inputdata& id){
    apta_node* n = aut->get_root();
    tail* t = tr->get_head(); // TODO: get
    for (int j = 0; j < t->get_length(); j++) {
        n = get_child_node(n, t);
        if (n == nullptr) {
            return nullptr; // Invariant: inputdata will never map to negative values.
        }
        t = t->future();
    }

    apta_node* ending_state = n;
    return ending_state;
}

apta_node* active_learning_namespace::get_last_node(const vector<int>& str, apta* aut, inputdata& id){
    trace* tr = vector_to_trace(str, id);
    return get_last_node(tr, aut, id);
}

/**
 * @brief Concatenates the two traces. Append tr2 to tr1. Takes the trace object of tr1 and appends the tails of tr2 on
 * it, if tr1 is a valid trace. If tr1 is uninitizalized (e.g. the access trace of the root node), is simply returns
 * tr2.
 *
 * @param tr1 The basic trace.
 * @param tr2 The trace we append to it.
 * @return trace* The resulting trace.
 */
trace* active_learning_namespace::concatenate_traces(trace* tr1, trace* tr2) {
    trace* res;
    if (tr1->get_length() == -1) {
        // the access trace of the root node
        res = mem_store::create_trace(inputdata_locator::get(), tr2);
        return res;
    }

    res = mem_store::create_trace(inputdata_locator::get(), tr1);
    if (res->is_finalized()) {
        tail* end_tail = res->end_tail;
        mem_store::delete_tail(end_tail);
    }

    // TODO: do we create a memory-leak with our creation of traces here?
    tail* parser = tr2->head;
    while (parser != nullptr) {
        tail* nt = mem_store::create_tail(parser);
        res->end_tail->set_future(nt);
        res->end_tail = nt;
    }

    if (!res->is_finalized()) {
        res->finalize();
    }

    return res;
}

/**
 * @brief This is the other version of the function. This one uses the types of the traces,
 * i.e. it implements accepting and rejecting traces. Hence we get a different case, and this is
 * the version that we use for e.g. L* and L#.
 *
 * @param tr The trace.
 * @param eval The evaluation function. Must inherit from count_driven.
 * @return true Accepts trace.
 * @return false Does not.
 */
bool active_learning_namespace::aut_accepts_trace(trace* tr, apta* aut, const count_driven* const eval) {
    const int trace_type = tr->get_type(); // eval->predict_type(tr);

    apta_node* n = aut->get_root();
    tail* t = tr->get_head();
    for (int j = 0; j < t->get_length(); j++) {
        n = active_learning_namespace::get_child_node(n, t);
        if (n == nullptr)
            return false; // TODO: does this one make sense?

        t = t->future();
    }

    if (trace_type == n->get_data()->predict_type(t))
        return true;
    return false;
}

/**
 * @brief This function is like the greedyrun method, but it additionally returns the refinements done.
 * We need this in active learning that whenever the equivalence oracle does not work out we will be able
 * to undo all the refinments and pose a fresh hypothesis later on.
 *
 * @param aut The apta.
 * @return list<refinement*> list with the refinements done.
 */
void active_learning_namespace::minimize_apta(list<refinement*>& refs, state_merger* merger) {
    refinement* top_ref = merger->get_best_refinement();
    while (top_ref != 0) {
        refs.push_back(top_ref);
        top_ref->doref(merger);
        top_ref = merger->get_best_refinement();
    }
}

/**
 * @brief Using the red-blue framework, build an automaton layer by layer and force the last layer to merge in if
 * there are inconsistencies between merges. This will help us to find early hypothesis for some of the algorithms,
 * e.g. pL#.
 *
 * Note: Currently only works with string_probability_estimator. In case other heuristics need to be employed we need to
 * do that in the heuristics (evaluation function) section.
 *
 * @param refs (Out): List with refinements that will be updated. Refinements performed in this function will be
 * appended to the list.
 * @param aut The unmerged apta/observation tree.
 */
void active_learning_namespace::find_closed_automaton(list<refinement*>& performed_refs, unique_ptr<apta>& aut,
                                                      unique_ptr<state_merger>& merger,
                                                      double (*distance_func)(apta*, apta_node*, apta_node*)) {
    while (true) {

        unordered_set<apta_node*> blue_nodes;
        for (blue_state_iterator b_it = blue_state_iterator(aut->get_root()); *b_it != nullptr; ++b_it) {
            auto blue_node = *b_it;
            blue_nodes.insert(blue_node);
        }
        if (blue_nodes.size() == 0)
            return;

        for (auto blue_node : blue_nodes) {
            if (blue_node->has_child_nodes()) {
                refinement_set possible_merges;
                for (red_state_iterator r_it = red_state_iterator(aut->get_root()); *r_it != nullptr; ++r_it) {
                    const auto red_node = *r_it;

                    refinement* ref = merger->test_merge(red_node, blue_node);
                    if (ref != nullptr) {
                        possible_merges.insert(ref);
                    }
                }
                if (possible_merges.size() == 0) {
                    refinement* ref = mem_store::create_extend_refinement(merger.get(), blue_node);
                    ref->doref(merger.get());
                    performed_refs.push_back(ref);
                } else {
                    // get the best refinement from the heap
                    refinement* best_merge = *(possible_merges.begin());
                    for (auto it : possible_merges) {
                        if (it != best_merge)
                            it->erase();
                    }
                    best_merge->doref(merger.get());
                    performed_refs.push_back(best_merge);
                }
            } else {
                // force a merge with the red node that introduces minimal error
                pair<apta_node*, double> best_node = make_pair(nullptr, std::numeric_limits<double>::max());
                for (red_state_iterator r_it = red_state_iterator(aut->get_root()); *r_it != nullptr; ++r_it) {
                    const auto red_node = *r_it;
                    double err = distance_func(aut.get(), red_node, blue_node);
                    if (err < best_node.second) {
                        best_node = make_pair(red_node, err);
                    }
                }

                refinement* ref =
                    mem_store::create_merge_refinement(merger.get(), best_node.second, best_node.first, blue_node);
                ref->doref(merger.get());
                performed_refs.push_back(ref);
            }
        }
    }
}

/**
 * @brief Resets the apta.
 *
 * Side effect: Exhausts the refs-stack to zero.
 *
 * @param merger The state merger.
 * @param refs List with refinements (a list so we can parse in reverse).
 */
void active_learning_namespace::reset_apta(state_merger* merger, const list<refinement*>& refs) {
    for (auto it = refs.rbegin(); it != refs.rend(); ++it) {
        const auto top_ref = *it;
        top_ref->undo(merger);
    }
}

/**
 * @brief Inverse operation to reset_apta. Does all the operations in refs in order again.
 * 
 * We need this sometimes, because some of the operations in flexfringe demand an unmerged apta, such as for example when 
 * we add traces to the apta, i.e. when we create new states. Used for example in L# and its derivatives.
 * 
 * @param merger The state merger.
 * @param refs List with refinements.
 */
void active_learning_namespace::do_operations(state_merger* merger, const std::list<refinement*>& refs){
    for (auto it = refs.begin(); it != refs.end(); ++it) {
        const auto top_ref = *it;
        top_ref->doref(merger);
    }
}


void active_learning_namespace::update_tail(tail* t, const int symbol) {
    static int num_tails = 0;

    auto td = t->td;
    td->symbol = symbol;
    // td->data = ""; // TODO: does not work yet with attributes
    td->tail_nr = num_tails++;

    int num_symbol_attributes = 0; // inputdata::get_num_symbol_attributes();
    if (num_symbol_attributes > 0) {
        cout << "This branch is not implemented yet" << endl;
    }
}

/**
 * @brief Add the sequence as a concatenation of tail-objects to the trace, so that flexfringe can work it out.
 *
 * @param new_trace The trace to add to.
 * @param sequence Sequence in list for.
 */
void active_learning_namespace::add_sequence_to_trace(trace* new_trace, const vector<int> sequence) {
    new_trace->length = sequence.size();

    tail* new_tail = mem_store::create_tail(nullptr);
    new_tail->tr = new_trace;
    new_trace->head = new_tail;

    for (int index = 0; index < sequence.size(); ++index) {
        const int symbol = sequence[index];
        active_learning_namespace::update_tail(new_tail, symbol);
        new_tail->td->index = index;

        tail* old_tail = new_tail;
        new_tail = mem_store::create_tail(nullptr);
        new_tail->tr = new_trace;
        old_tail->set_future(new_tail);
    }

    new_tail->td->index = sequence.size();
    new_trace->end_tail = new_tail;

    new_trace->finalize();
}

/**
 * @brief Turns a list to a trace.
 *
 * @param vec The list.
 * @param id The inputdata.
 * @param trace_type Accepting or rejecting.
 * @return trace* The trace.
 */
trace* active_learning_namespace::vector_to_trace(const vector<int>& vec, inputdata& id, const int trace_type) {
    static int trace_nr = 0;

    trace* new_trace = mem_store::create_trace(&id);

    new_trace->type = trace_type;
    new_trace->sequence = ++trace_nr;

    active_learning_namespace::add_sequence_to_trace(new_trace, vec);

    return new_trace;
}

/**
 * @brief Gets the probability of the current last symbol represented by the trace.
 *
 * Warning: This function will not work on the empty string, so as for the root node, you'll have to treat it
 * separately.
 *
 * TODO: you can optimize this guy via splitting into two functions. One to get probability to the current state, and
 * then you can ask the probability for each outgoing state. Will save quite a bit of computations.
 *
 * @param tr The trace.
 * @param id The inputdata.
 * @return const double Probability.
 */
/*const double active_learning_namespace::get_probability_of_last_symbol(trace* tr, inputdata& id,
                                                                       const unique_ptr<base_oracle>& oracle,
                                                                       apta* aut) {
    static unordered_map<apta_node*, unordered_map<int, double>> node_response_map; // memoization

    apta_node* n = aut->get_root();
    tail* t = tr->head;
    pref_suf_t
        current_string; // TODO: constructing a fresh vector is possibly inefficient. can we walk around this guy here?

    double product_probability = 1;
    while (n != nullptr) {
        auto symbol = t->get_symbol();
        current_string.push_back(symbol);

        if (t->future()->is_final()) {
            // the magic happens here
            if (!node_response_map.contains(n))
                node_response_map[n] = unordered_map<int, double>();
            else if (node_response_map[n].contains(symbol))
                return node_response_map[n][symbol];

            double new_p = oracle->ask_sul(current_string, id).get_double();
            if (new_p == 0) {
                node_response_map[n][symbol] = 0;
                return 0;
            }

            auto res = new_p / product_probability;
            node_response_map[n][symbol] = res;
            return res;
        }

        if (!node_response_map.contains(n))
            node_response_map[n] = unordered_map<int, double>();
        else if (node_response_map[n].contains(symbol)) {
            product_probability *= node_response_map[n][symbol];
            if (product_probability == 0)
                return 0;
        } else {
            double new_p = oracle->ask_sul(current_string, id).get_double();
            node_response_map[n][symbol] = new_p / product_probability;
            product_probability *= new_p;
            if (product_probability == 0)
                return 0;
        }

        n = active_learning_namespace::get_child_node(n, t);
        t = t->future();
    }

    throw runtime_error("We should not reach here. What happened?");
}*/

void active_learning_namespace::print_span(std::span<const int> l) {
    for (const auto s : l) cout << s << " ";
    cout << endl;
}
