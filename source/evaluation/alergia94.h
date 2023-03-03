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
    virtual double compute_score(state_merger*, apta_node* left, apta_node* right);
    virtual bool pool_and_compute_tests(num_map& left_map, int left_total, int left_final, num_map& right_map, int right_total, int right_final);
};

#endif
