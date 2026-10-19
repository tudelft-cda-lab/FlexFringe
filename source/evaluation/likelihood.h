#ifndef __LIKELIHOOD__
#define __LIKELIHOOD__

#include "alergia.h"

/* The data contained in every node of the prefix tree or DFA */
class likelihood_data: public alergia_data {
protected:
  REGISTER_DEC_DATATYPE(likelihood_data);
public:
    double undo_loglikelihood_orig;
    double undo_loglikelihood_merged;
    int undo_extra_parameters;

    likelihood_data();
    virtual void initialize();
};

class likelihoodratio: public alergia {

protected:
  REGISTER_DEC_TYPE(likelihoodratio);
  

public:
  void update_likelihood(double,double,double,double);

  double loglikelihood_orig;
  double loglikelihood_merged;
  double extra_parameters;

  virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right, int depth);
  virtual void update_score(state_merger *merger, apta_node* left, apta_node* right);
  virtual bool compute_consistency(state_merger *merger, apta_node* left, apta_node* right);
  virtual double  compute_score(state_merger*, apta_node* left, apta_node* right);
  virtual void reset(state_merger *merger);

    virtual void split_update_score_before(state_merger*, apta_node* left, apta_node* right, tail* t);
    virtual void split_update_score_after(state_merger*, apta_node* left, apta_node* right, tail* t);

    virtual bool split_compute_consistency(state_merger *, apta_node* left, apta_node* right);
    virtual double split_compute_score(state_merger *, apta_node* left, apta_node* right);

    void update_likelihood_pool(double left_count, double right_count, double left_divider, double right_divider);

    void delete_likelihood(double left_count, double right_count, double left_divider, double right_divider);

    void delete_likelihood_pool(double left_count, double right_count, double left_divider, double right_divider);

    virtual double compute_global_score(state_merger *);
    virtual double compute_partial_score(state_merger *);
};

#endif
