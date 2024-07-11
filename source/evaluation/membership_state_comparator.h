/**
 * @file membership_state_comparator.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-06-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef __MEMBERSHIP_STATE_COMPARATOR__
#define __MEMBERSHIP_STATE_COMPARATOR__

#include "lsharp_eval.h"

#include <unordered_map>
#include <optional>

class membership_state_comparator_data: public lsharp_data {
    friend class membership_state_comparator;
protected:

    REGISTER_DEC_DATATYPE(membership_state_comparator_data);

    int N; // number of times we updated the stats
    std::vector<float> LS; // used to compute floating statistics
    std::vector<float> SS; // used to compute floating statistics

    std::vector<float> means;
    std::vector<float> std_devs;

public:
    membership_state_comparator_data() : lsharp_data::lsharp_data(){     
        N = 0;   
    }

    //virtual void print_transition_label(iostream& output, int symbol) override;
    virtual void print_state_label(iostream& output) override;

    virtual void update(evaluation_data* right) override;
    virtual void undo(evaluation_data* right) override;

    void update_sums(const std::vector<float>& internal_rep);
    void compute_statistics();
};

class membership_state_comparator: public lsharp_eval {

protected:
    REGISTER_DEC_TYPE(membership_state_comparator);

    float score;

public:
    float get_diff(const vector<float>& left_v, const vector<float>& right_v) const;
    virtual bool consistent(state_merger *merger, apta_node* left_node, apta_node* right_node, int depth) override;

    //virtual double compute_score(state_merger*, apta_node* left_node, apta_node* right_node) override;
    virtual void reset(state_merger *merger) override;

    //static double get_distance(apta* aut, apta_node* left_node, apta_node* right_node);
};

#endif
