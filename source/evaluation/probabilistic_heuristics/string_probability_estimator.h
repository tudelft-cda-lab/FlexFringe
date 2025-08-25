/**
 * @file string_probability_estimator.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-09-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __STRING_PROBABILITY_ESTIMATOR__
#define __STRING_PROBABILITY_ESTIMATOR__

#include "evaluate.h"
#include "probabilistic_heuristic_interface.h"

#include <unordered_map>
#include <optional>

/**
 * @brief Evaluation function data supporting the operations from the paper 
 * "PDFA Distillation via String Probability Queries", Baumgartner and Verwer 2024.
 * 
 */
class string_probability_estimator_data: public probabilistic_heuristic_interface_data {
    friend class string_probability_estimator;
protected:

    REGISTER_DEC_DATATYPE(string_probability_estimator_data);
    
    std::vector<double> seen_probability_mass; // for merges and actual predictions

public:
    void print_transition_label(std::iostream& output, int symbol) override;
    void print_state_label(std::iostream& output) override;

    void update(evaluation_data* right) override;
    void undo(evaluation_data* right) override;

    void read_json(json& node) override;
    void write_json(json& node) override;

    int predict_symbol(tail*) override;
    double predict_symbol_score(int t) override;

    void add_probability(const int symbol, const double p);
    void update_probability(const int symbol, const double p);

    void init_access_probability(const double p) noexcept override {
        this->access_trace_prob = p;
        this->final_prob = p;
    }

    void initialize_distributions(const std::vector<int>& alphabet){
        seen_probability_mass.resize(alphabet.size());
        symbol_probability_map.resize(alphabet.size());
        for(int i=0; i < alphabet.size(); ++i){ // TODO: zero-initialization makes this one obsolete?
            seen_probability_mass[i] = 0;
            symbol_probability_map[i] = 0;
        }
    }

    /**
     * @brief Updates the final probability as described in "PDFA Distillation via String Probability Queries", 
     * Baumgartner and Verwer 2024.
     * 
     * @param product The product leading to the node [lambda(x)]. 
     * @param is_root True if the node is the root node.
     */
    void update_final_prob(const double product, const bool is_root=false){
        static const double mu = MU;

        if(!FINAL_PROBABILITIES){ // TODO: remove finalprob option. On by default
            this->final_prob = 1;
            return;
        }
        else if(is_root){
            this->final_prob = product;
            return;
        } 
        
        this->final_prob = std::min(this->access_trace_prob / product, 1.0 - mu); // TODO: min value here?
    }

    double get_non_normalized_probability_mass(const int symbol) {
        return seen_probability_mass[symbol];
    }

    // TODO: replace with get_probabilities
    /* std::vector<double>& get_outgoing_distribution(){
        return symbol_probability_map;
    } */

    std::vector<double>& get_non_normalized_distribution(){
        return seen_probability_mass;
    }
};

class string_probability_estimator: public probabilistic_heuristic_interface {

protected:
    REGISTER_DEC_TYPE(string_probability_estimator);

    double score;

public:

    virtual bool consistent(state_merger *merger, apta_node* left_node, apta_node* right_node, int depth) override;

    virtual double compute_score(state_merger*, apta_node* left_node, apta_node* right_node) override;
    virtual void reset(state_merger *merger) override;

    static void add_outgoing_probs(apta_node* node, std::unordered_map<int, double>& probabilities);
    static void normalize_probabilities(string_probability_estimator_data* data);
};

#endif
