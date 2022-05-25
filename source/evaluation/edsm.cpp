#include <stdlib.h>
#include <math.h>
#include <vector>
#include <map>
#include <stdio.h>
//#include <gsl/gsl_cdf.h>

#include "state_merger.h"
#include "evaluate.h"
#include "edsm.h"

#include "parameters.h"

REGISTER_DEF_DATATYPE(edsm_data);
REGISTER_DEF_TYPE(evidence_driven);

/* Evidence driven state merging, count number of pos-pos and neg-neg merges */
void evidence_driven::update_score(state_merger *merger, apta_node* left, apta_node* right){
    edsm_data* l = (edsm_data*) left->get_data();
    edsm_data* r = (edsm_data*) right->get_data();

    if(l->pos_final() > 0 && r->pos_final() > 0) num_pos += 1;
    if(l->neg_final() > 0 && r->neg_final() > 0) num_neg += 1;
};

double evidence_driven::compute_score(state_merger *merger, apta_node* left, apta_node* right){
  return num_pos + num_neg;
};

void evidence_driven::reset(state_merger *merger){
  inconsistency_found = false;
  num_pos = 0;
  num_neg = 0;
};
