/**
 * @file common_functions.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-02-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _COMMON_FUNCTIONS_H_
#define _COMMON_FUNCTIONS_H_

#include "apta.h"
#include "count_types.h"
#include "definitions.h"
#include "evaluate.h"
#include "input/inputdata.h" // why does it not find it?
#include "parameters.h"
#include "refinement.h"
#include "input/tail.h" // why does it not find it?
#include "input/trace.h"

#include <functional>
#include <list>
#include <unordered_map>
#include <utility>

namespace {
    template<typename T>
    concept less_than = requires(T t1, T t2){
        t1 < t2;
    };
};

namespace active_learning_namespace {

    apta_node* get_child_node(apta_node* n, tail* t);
    apta_node* get_child_node(apta_node* n, int symbol);
    bool aut_accepts_trace(trace* tr, apta* aut);
    bool aut_accepts_trace(trace* tr, apta* aut, const count_driven* const eval);
    int count_nodes(apta* aut);

    const int predict_type_from_trace(trace* tr, apta* aut, inputdata& id);

    apta_node* get_last_node(trace* tr, apta* aut, inputdata& id);
    apta_node* get_last_node(const std::vector<int>& str, apta* aut, inputdata& id);

    trace* concatenate_traces(trace* tr1, trace* tr2);

    void reset_apta(state_merger* merger, const std::list<refinement*>& refs);
    void do_operations(state_merger* merger, const std::list<refinement*>& refs);
    void minimize_apta(std::list<refinement*>& refs, state_merger* merger);
    void find_closed_automaton(std::list<refinement*>& performed_refs, std::unique_ptr<apta>& aut,
                                            std::unique_ptr<state_merger>& merger,
                                            double (*distance_func)(apta*, apta_node*, apta_node*));

    template<typename T>
    std::vector<T> concatenate_vectors(const std::vector<T>& prefix, const std::vector<T>& suffix){
        std::vector<T> res;
        res.reserve(prefix.size() + suffix.size());
        res.insert(res.end(), prefix.begin(), prefix.end());
        res.insert(res.end(), suffix.begin(), suffix.end());    
        return res;
    }

    trace* vector_to_trace(const std::vector<int>& vec, inputdata& id, const int trace_type = 0);

    void add_sequence_to_trace(/*out*/ trace* new_trace, const std::vector<int> sequence);
    void update_tail(/*out*/ tail* t, const int symbol);


    const double get_sampled_probability(
        trace* tr, inputdata& id, apta* aut,
        std::shared_ptr<std::unordered_map<apta_node*, std::unordered_map<int, int>>>& node_type_counter);

    /**
     * @brief Compares reference-wrappers of a type.
     *
     */
    template <less_than T>
    struct ref_wrapper_comparator {
    public:
        ref_wrapper_comparator() = default;

        constexpr bool operator()(const std::reference_wrapper<T>& left, const std::reference_wrapper<T>& right) const {
            return left.get() < right.get();
        }
    };

    /**
     * @brief For debugging
     */
    template <class it_T> [[maybe_unused]] void print_sequence(it_T begin, it_T end) {
        std::cout << "seq: ";
        for (; begin != end; ++begin) std::cout << *begin << " ";
        std::cout << std::endl;
    }

    [[maybe_unused]] void print_list(const std::list<int>& l);

    void print_vector(const std::vector<int>& l);
} // namespace active_learning_namespace


#endif
