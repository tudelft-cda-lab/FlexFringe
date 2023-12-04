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

#include <unordered_map>
#include <optional>

class string_probability_estimator_data: public evaluation_data {
    friend class string_probability_estimator;
protected:

    REGISTER_DEC_DATATYPE(string_probability_estimator_data);

    double final_prob;
    double access_trace_prob; // this is the total string probability to end in this state. We need it so that we can compute the final prob
    
    std::vector<double> symbol_probability_map;
    std::vector<double> normalized_symbol_probability_map; // for merges and actual predictions

public:
    string_probability_estimator_data() : evaluation_data::evaluation_data(){        
        final_prob = 0;
    }
    virtual void print_transition_label(iostream& output, int symbol) override;
    virtual void print_state_label(iostream& output) override;

    virtual void update(evaluation_data* right) override;
    virtual void undo(evaluation_data* right) override;

    virtual void read_json(json& node) override;
    virtual void write_json(json& node) override;

    virtual int predict_symbol(tail*) override;
    virtual double predict_symbol_score(int t) override;

    void add_probability(const int symbol, const double p);
    void update_probability(const int symbol, const double p);

    void initialize_distributions(const vector<int>& alphabet){
        symbol_probability_map.resize(alphabet.size());
        normalized_symbol_probability_map.resize(alphabet.size());
        for(int i=0; i < alphabet.size(); ++i){
            symbol_probability_map[i] = 0;
            normalized_symbol_probability_map[i] = 0;
        }
    }

    void init_access_probability(const double p) noexcept {
        this->access_trace_prob = p;
        this->final_prob = p;
    }

    double get_final_prob() noexcept {
        return this->final_prob;
    }

    /**
     * @brief TODO: update this function
     * 
     * @param product The product leading to the node [lambda(x)]. 
     * @param is_root True if the node is the root node.
     * 
     * @return true Computed final prob larger than 1.
     * @return false Final prob smaller than or equal 1.
     */
    /* bool */ void update_final_prob(const double product, const bool is_root=false){
        static const double mu = MU;

        if(!FINAL_PROBABILITIES){ // TODO: remove finalprob option. On by default
            this->final_prob = 1;
            return /* false */;
        }
        
        if(is_root){
            this->final_prob = product;
            return /* false */;
        } 
        
        this->final_prob = min(this->access_trace_prob / product, 1.0 - mu); // TODO: min value here?
        /* return final_prob > 1; */
    }

    double get_normalized_probability(const int symbol) {
        return normalized_symbol_probability_map[symbol];
    }

    std::vector<double>& get_outgoing_distribution(){
        return symbol_probability_map;
    }

    std::vector<double>& get_normalized_distribution(){
        return normalized_symbol_probability_map;
    }
};

class string_probability_estimator: public evaluation_function {

protected:
    REGISTER_DEC_TYPE(string_probability_estimator);

    double score;

public:

    virtual bool consistent(state_merger *merger, apta_node* left_node, apta_node* right_node, int depth) override;

    virtual double compute_score(state_merger*, apta_node* left_node, apta_node* right_node) override;
    virtual void reset(state_merger *merger) override;

    static void add_outgoing_probs(apta_node* node, std::unordered_map<int, double>& probabilities);
    static void normalize_probabilities(string_probability_estimator_data* data);

    static double get_distance(apta* aut, apta_node* left_node, apta_node* right_node);

    //static void update_mu(apta_node* node); // TODO: implement
};

#endif
