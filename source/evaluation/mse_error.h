#ifndef __MSEERROR__
#define __MSEERROR__

#include "evaluate.h"

typedef std::list<double> double_list;

/* The data contained in every node of the prefix tree or DFA */
class mse_data: public evaluation_data {

protected:
  REGISTER_DEC_DATATYPE(mse_data);
  
public:
    double undo_rss_before;
    double undo_rss_after;
    double undo_num_points;
    double undo_total_merges;

  /* occurences of this state */
    double num_tails;
    std::vector<double> sums;
    std::vector<double> sum_squares;

  void print_state_label(std::iostream &output);

  void read_json(json &data);

  void write_json(json &data);

    mse_data();

  virtual void initialize();


    virtual void add_tail(tail* t);

  void del_tail(tail *t);

  virtual void update(evaluation_data* right);
    virtual void undo(evaluation_data* right);

  double predict_data_score(tail *t);

  double predict_data_score(std::string s);

  std::string predict_data(tail*);
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
  
  virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right);
  virtual void update_score(state_merger *merger, apta_node* left, apta_node* right);
  virtual double  compute_score(state_merger*, apta_node* left, apta_node* right);

  bool compute_consistency(state_merger *merger, apta_node *left, apta_node *right);

  virtual void reset(state_merger *merger);

  void split_update_score_before(state_merger *merger, apta_node *left, apta_node *right, tail *t);

  void split_update_score_after(state_merger *merger, apta_node *left, apta_node *right, tail *t);

  bool split_compute_consistency(state_merger*, apta_node *left, apta_node *right);

  double split_compute_score(state_merger*, apta_node *left, apta_node *right);

  virtual int sink_type(apta_node* node);
  virtual bool sink_consistent(apta_node* node, int type);
  virtual int num_sink_types();
};

#endif
