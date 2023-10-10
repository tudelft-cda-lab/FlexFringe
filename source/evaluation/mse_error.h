#ifndef __MSEERROR__
#define __MSEERROR__

#include "evaluate.h"

typedef list<double> double_list;

/* The data contained in every node of the prefix tree or DFA */
class mse_data: public evaluation_data {

protected:
  REGISTER_DEC_DATATYPE(mse_data);
  
public:
    /* occurences of this state */
    double_list occs;
    double mean;
    double_list::iterator merge_point;

    mse_data();

    virtual void add_tail(tail* t);
    virtual void update(evaluation_data* right);
    virtual void undo(evaluation_data* right);

    virtual string predict_data(tail *);
};

class mse_error: public evaluation_function{

protected:
  REGISTER_DEC_TYPE(mse_error);
  
  state_set aic_states;

public:
  double num_merges = 0;
  double num_points = 0;
  double RSS_before = 0.0;
  double RSS_after = 0.0;
  int total_merges = 0;
  double prev_AIC = 0;
  
  virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right, int depth);
  virtual void data_update_score(mse_data* l, mse_data* r);
  virtual void update_score(state_merger *merger, apta_node* left, apta_node* right);
  virtual double  compute_score(state_merger*, apta_node* left, apta_node* right);
  virtual void reset(state_merger *merger);

  virtual int sink_type(apta_node* node);
  virtual bool sink_consistent(apta_node* node, int type);
  virtual int num_sink_types();
};

#endif
