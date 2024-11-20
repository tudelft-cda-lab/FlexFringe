/**
 * @file probabilistic_heuristic_interface.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-11-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __PROBABILISTIC_HEURISTIC_INTERFACE__
#define __PROBABILISTIC_HEURISTIC_INTERFACE__

#include "evaluation.h"

#include <type_traits>

class probabilistic_heuristic_interface_data : public evaluation_data {
protected:
    double final_prob;
    double access_trace_prob; // this is the total string probability to end in this state. We need it so that we can compute the final prob
    
    typename std::vector<double> symbol_probability_map;

public:
    probabilistic_heuristic_interface_data() = delete;

    double get_final_probability() noexcept {
        return this->final_prob;
    }

    void set_final_probability(const double prob){
        this->final_prob = prob;
    }

    virtual void init_access_probability(const double p) noexcept {
        this->access_trace_prob = p;
    }

    double get_access_probability() const noexcept {
        return this->access_prob;
    }

    const std::vector<double>& get_probabilities() noexcept {
        return symbol_probability_map;
    }

    double get_probability(const int symbol) noexcept {
        return symbol_probability_map[symbol];
    }

    void set_probability(const int symbol, const double p) noexcept {
        symbol_probability_map[symbol] = p;
    }
};

class probabilistic_heuristic_interface : evaluation_function {
public:
    probabilistic_heuristic_interface() = delete;
    static double get_merge_distance_access_trace(apta* aut, apta_node* left_node, apta_node* right_node);
};

#endif
