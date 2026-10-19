/**
 * @file lstar_evaluation.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Evaluation function of L* algortihm. Checks if outgoing strings are equal, else false.
 * @version 0.1
 * @date 2023-03-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __L_STAR_EVALUATION__
#define __L_STAR_EVALUATION__

#include "evaluate.h"

/* The data contained in every node of the prefix tree or DFA */
class lstar_evaluation_data: public evaluation_data {

protected:

    REGISTER_DEC_DATATYPE(lstar_evaluation_data);

public:
    virtual void update(evaluation_data* right){}; // no need for those, works directly on prefix tree
    virtual void undo(evaluation_data* right){}; // no need for those, works directly on prefix tree
};

class lstar_evaluation: public evaluation_function {

protected:
    REGISTER_DEC_TYPE(lstar_evaluation);

public:

    virtual bool consistent(state_merger *merger, apta_node* left, apta_node* right, int depth){return true;}; // TODO
    virtual double compute_score(state_merger*, apta_node* left, apta_node* right){return 0;}; // TODO
    virtual void reset(state_merger *merger){}; // TODO
};

#endif
