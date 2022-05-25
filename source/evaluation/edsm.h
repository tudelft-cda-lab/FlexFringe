#ifndef __EVIDENCE__
#define __EVIDENCE__

#include "count_types.h"

/* The data contained in every node of the prefix tree or DFA */
class edsm_data: public count_data {

protected:
  REGISTER_DEC_DATATYPE(edsm_data);

public:
};

class evidence_driven: public count_driven {

protected:
  REGISTER_DEC_TYPE(evidence_driven);

public:
  int num_pos;
  int num_neg;

  virtual void update_score(state_merger *merger, apta_node* left, apta_node* right);
  virtual double  compute_score(state_merger*, apta_node* left, apta_node* right);
  virtual void reset(state_merger *merger);
};

#endif
