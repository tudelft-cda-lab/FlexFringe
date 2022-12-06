#ifndef __RTIPLUS__
#define __RTIPLUS__

#include "likelihood.h"

typedef vector< vector<double> > quantile_map;

/* The data contained in every node of the prefix tree or DFA */
class rtiplus_data: public likelihood_data {
protected:
  REGISTER_DEC_DATATYPE(rtiplus_data);

public:

    quantile_map statistics;
    
    rtiplus_data();

    virtual void initialize();
    
    virtual void add_tail(tail* t);
    virtual void update(evaluation_data* right);
    virtual void undo(evaluation_data* right);
    
    virtual void split_update(evaluation_data* other);
    virtual void split_undo(evaluation_data* other);
    
    virtual void del_tail(tail* t);

    virtual void print_state_label(iostream& output);

    virtual void read_json(json& node);
    virtual void write_json(json& node);

    virtual double predict_score(tail* t);
};

class rtiplus: public likelihoodratio {

protected:
  REGISTER_DEC_TYPE(rtiplus);
  
public:
    
  static vector< vector<double> > attribute_quantiles;

  virtual void update_score(state_merger *merger, apta_node* left, apta_node* right);
  //virtual void update_score_after(state_merger *merger, apta_node* left, apta_node* right);

  virtual void split_update_score_before(state_merger*, apta_node* left, apta_node* right, tail* t);
  virtual void split_update_score_after(state_merger*, apta_node* left, apta_node* right, tail* t);
    
  virtual bool split_compute_consistency(state_merger *, apta_node* left, apta_node* right);
  virtual double split_compute_score(state_merger *, apta_node* left, apta_node* right);
    
  virtual void initialize_after_adding_traces(state_merger* merger);
  virtual void initialize_before_adding_traces();
  virtual void reset_split(state_merger *, apta_node *);
};

#endif
