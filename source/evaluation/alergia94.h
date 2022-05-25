#ifndef __ALERGIA94__
#define __ALERGIA94__

#include "evaluate.h"
#include "alergia.h"

typedef map<int, int> num_map;

/* The data contained in every node of the prefix tree or DFA */
class alergia94_data: public alergia_data {

protected:
  REGISTER_DEC_DATATYPE(alergia94_data);

public:
    alergia94_data();
};


class alergia94: public alergia {

protected:
  REGISTER_DEC_TYPE(alergia94);

public:
  static bool alergia_consistency(double right_count, double left_count, double right_total, double left_total);
  static int EVAL_TYPE;

  virtual bool data_consistent(alergia94_data* l, alergia94_data* r);
  virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right);
};

#endif
