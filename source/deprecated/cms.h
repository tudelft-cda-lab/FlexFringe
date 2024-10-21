/**
 * @file cms.h
 * @author Robert Baumgartner, Raffail Skoulos
 * @brief This file implements count-min-sketch based state merging.
 * @version 0.1
 * @date 2020-05-29
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef __CMS__
#define __CMS__

#include "alergia.h"
#include "countminsketch.h"

#include <string>

/* The data contained in every node of the prefix tree or DFA */
class cms_data : public alergia_data {
protected:
    REGISTER_DEC_DATATYPE(cms_data);

    const std::vector<int> get_n_grams(tail* t, const int n_steps) const;
    std::vector< CountMinSketch<int> > sketches;

public:

    cms_data();

    inline const std::vector< CountMinSketch<int> >& get_sketches() const {return this->sketches; }

    virtual void update(evaluation_data *right) override;
    virtual void undo(evaluation_data *right) override;

    void add_tail(tail* t) override;

    //************************************* Sinks **********************************************

    inline bool is_low_count_sink() const noexcept {
        return this->node->get_size() < SINK_COUNT;
    }

    //inline bool is_stream_sink(apta_node* node) const noexcept {
    //    return this->node->get_size() < STREAM_COUNT;
    //}

    inline int sink_type(apta_node* node) const noexcept {
        if(!USE_SINKS) return -1;

        if (is_low_count_sink()) return 0;
        //if(is_stream_sink(node)) return 2;
        return -1;
    };

    inline bool sink_consistent(int type) const noexcept {
        if(!USE_SINKS) return true;

        if(type == 0) return is_low_count_sink();
        return true;
    };

    inline int num_sink_types() const noexcept{
        if(!USE_SINKS) return 0;
        return 2;
    };

    void print_state_label(std::iostream& output);
};

class cms : public alergia {

protected:
    REGISTER_DEC_TYPE(cms);
    double scoresum = 0;

public:
    virtual double compute_score(state_merger *, apta_node *left, apta_node *right);

    virtual void reset(state_merger *merger);

    virtual bool consistent(state_merger *merger, apta_node *left, apta_node *right);
    //virtual void initialize(state_merger *);

};

#endif
