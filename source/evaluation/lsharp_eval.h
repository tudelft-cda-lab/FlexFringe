#ifndef __LSHARP_EVAL__
#define __LSHARP_EVAL__

#include "count_types.h"

class lsharp_data: public count_data {

protected:
  REGISTER_DEC_DATATYPE(lsharp_data);

public:
};

class lsharp_eval: public count_driven {

protected:
  REGISTER_DEC_TYPE(lsharp_eval);

public:
  bool consistent(state_merger *merger, apta_node* left, apta_node* right, int depth) override;
  double compute_score(state_merger* merger, apta_node* left, apta_node* right) override;
};

#endif
