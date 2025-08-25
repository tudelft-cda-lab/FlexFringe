#ifndef __MEALY__
#define __MEALY__

#include "evaluate.h"

class mealy_data;

typedef std::map<int, int> output_map;
typedef std::map<int, mealy_data*> undo_map;
typedef std::map<int, std::string> is_map;
typedef std::map<std::string, int> si_map;

/* The data contained in every node of the prefix tree or DFA */
class mealy_data: public evaluation_data {

protected:
  REGISTER_DEC_DATATYPE(mealy_data);

public:
    output_map  outputs;
    undo_map    undo_info;
    
    static int num_outputs;
    static si_map output_int;
    static is_map int_output;
    
    virtual void add_tail(tail* t);
    virtual void update(evaluation_data* right);
    virtual void undo(evaluation_data* right);

    void print_transition_label(std::iostream&, int);

    virtual std::string predict_data(tail*);
};

class mealy: public evaluation_function {

protected:
  REGISTER_DEC_TYPE(mealy);

public:

  int num_matched;
  int num_unmatched;

  virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right, int depth);
  //virtual void print_dot(iostream&, state_merger *);
  virtual bool compute_consistency(state_merger *, apta_node* left, apta_node* right);
  virtual double  compute_score(state_merger*, apta_node* left, apta_node* right);
  virtual void reset(state_merger *);

  int num_sink_types();

};

#endif
