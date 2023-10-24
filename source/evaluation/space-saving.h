/**
 * @file space-saving.h
 * @author Robert Baumgartner
 * @brief This file implements Balle et al.'s merge heuristic (Bootstrapping and learning PDFA from data streams, 2012)
 * @version 0.1
 * @date 2020-05-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __SPACE_SAVING__
#define __SPACE_SAVING__

#include "alergia.h"
#include "prefspacesavesketch.h"

#include <string>
#include <vector>

/* The data contained in every node of the prefix tree or DFA */
class space_saving_data : public alergia_data {
protected:
    REGISTER_DEC_DATATYPE(space_saving_data);

    PrefSpSvSketch main_sketch;
    vector<PrefSpSvSketch> bootstrapped_sketches;
public:

    space_saving_data();

    virtual void update(evaluation_data *right) override;
    virtual void undo(evaluation_data *right) override;

    void add_tail(tail* t) override;

    bool check_lower_bound(space_saving_data* other);
    double get_upper_bound(space_saving_data* other);

    // for experimental purposes, not in original work
    bool hoeffding_check(space_saving_data* other) const;
    double cosine_sim(space_saving_data* other) const;

    //************************************* Sinks **********************************************

    inline bool is_low_count_sink() const noexcept {
        return this->node->get_size() < SINK_COUNT;
    }

/*     inline bool is_stream_sink(apta_node* node) const noexcept {
        return this->node->get_size() < STREAM_COUNT;
    }

    inline int sink_type(apta_node* node) const noexcept {
        if(!USE_SINKS) return -1;

        if (is_low_count_sink()) return 0;
        if(is_stream_sink(node)) return 2;
        return -1;
    }; */

    inline bool sink_consistent(int type) const noexcept {
        if(!USE_SINKS) return true;

        if(type == 0) return is_low_count_sink();
        return true;
    };

    inline int num_sink_types() const noexcept{
        if(!USE_SINKS) return 0;
        return 2;
    };

    void print_state_label(iostream& output);
};

class space_saving : public alergia {

protected:
    REGISTER_DEC_TYPE(space_saving);
    double score = 0.;

public:
    virtual double compute_score(state_merger *, apta_node *left, apta_node *right);

    virtual void reset(state_merger *merger);

    virtual bool consistent(state_merger *merger, apta_node *left, apta_node *right, int depth) override;
    virtual void initialize_before_adding_traces();

};

#endif
