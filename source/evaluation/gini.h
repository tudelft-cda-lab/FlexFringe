#ifndef __GINI__
#define __GINI__

#include "edsm.h"

/* The data contained in every node of the prefix tree or DFA */
class gini_data: public edsm_data {

protected:
  REGISTER_DEC_DATATYPE(gini_data);

public:
};

class gini: public evidence_driven {

protected:
  REGISTER_DEC_TYPE(gini);
  double merge_score;
  double split_score;
  double num_split;
  double num_merge;

public:
  virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right);
  virtual void update_score(state_merger *merger, apta_node* left, apta_node* right);
  virtual bool compute_consistency(state_merger *merger, apta_node* left, apta_node* right);
  virtual double  compute_score(state_merger*, apta_node* left, apta_node* right);

  virtual void split_update_score_before(state_merger*, apta_node* left, apta_node* right, tail* t);
  virtual void split_update_score_after(state_merger*, apta_node* left, apta_node* right, tail* t);
  virtual bool split_compute_consistency(state_merger *, apta_node* left, apta_node* right);
  virtual double split_compute_score(state_merger *, apta_node* left, apta_node* right);

  virtual void reset(state_merger *merger);
};

#endif
