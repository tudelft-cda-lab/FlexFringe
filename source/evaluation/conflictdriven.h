#ifndef __CONFLICTDRIVEN__
#define __CONFLICTDRIVEN__

#include "counttypes.h"

typedef std::list<apta_node*> node_list;

/* The data contained in every node of the prefix tree or DFA */
class conflict_data: public count_data {
  public:
	  node_list conflicts;
    node_list::iterator merge_point;
};


class conflict_driven: public count_driven{

protected:
  static DerivedRegister<conflict_driven> reg;

public:
  virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right);
  virtual void update_score(state_merger *merger, apta_node* left, apta_node* right);
  virtual double  compute_score(state_merger*, apta_node* left, apta_node* right);
  virtual void reset(state_merger *merger);
  virtual bool compute_consistency(state_merger *merger, apta_node* left, apta_node* right);
};

#endif
