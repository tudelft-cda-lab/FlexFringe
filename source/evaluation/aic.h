#ifndef __AIKE__
#define __AIKE__

#include "evaluate.h"
#include "likelihood.h"

/* The data contained in every node of the prefix tree or DFA */
class aic_data: public likelihood_data {
protected:
  REGISTER_DEC_DATATYPE(aic_data);
};

class aic: public likelihoodratio{

protected:
  REGISTER_DEC_TYPE(aic);

public:
    virtual bool compute_consistency(state_merger *merger, apta_node* left, apta_node* right);
    virtual bool split_compute_consistency(state_merger *, apta_node *left, apta_node *right);
    virtual double compute_score(state_merger*, apta_node* left, apta_node* right);
    virtual double split_compute_score(state_merger *, apta_node *left, apta_node *right);
};

#endif
