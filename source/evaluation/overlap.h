#ifndef __OVERLAPDRIVEN__
#define __OVERLAPDRIVEN__

#include "alergia.h"

/* The data contained in every node of the prefix tree or DFA */
class overlap_data: public alergia_data {
protected:
  REGISTER_DEC_DATATYPE(overlap_data);

public:

  virtual void print_transition_label(iostream& output, int symbol);
  virtual void print_state_label(iostream &output);

    inline int pos(int symbol){
        return count(symbol);
    }

    inline num_map::iterator pos_begin(){
        return counts_begin();
    }

    inline num_map::iterator pos_end(){
        return counts_end();
    }
};

class overlap_driven: public alergia {

protected:
  REGISTER_DEC_TYPE(overlap_driven);

public:
  double overlap;

  virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right);
  virtual void update_score(state_merger *merger, apta_node* left, apta_node* right);
  virtual double  compute_score(state_merger*, apta_node* left, apta_node* right);
  virtual void reset(state_merger *merger);
};

#endif
