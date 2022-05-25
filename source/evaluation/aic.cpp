#include "state_merger.h"
#include "evaluate.h"
#include "aic.h"
#include "parameters.h"

REGISTER_DEF_DATATYPE(aic_data);
REGISTER_DEF_TYPE(aic);

/* Akaike Information Criterion (AIC), computes the AIC value and uses it as score, AIC increases are inconsistent */
bool aic::compute_consistency(state_merger *merger, apta_node* left, apta_node* right){
  if (inconsistency_found) return false;
  if (extra_parameters == 0) return false;
  return 2.0 * ( extra_parameters - (loglikelihood_orig - loglikelihood_merged) ) > CHECK_PARAMETER;
};

bool aic::split_compute_consistency(state_merger *, apta_node* left, apta_node* right){
    if (inconsistency_found) return false;
    if (extra_parameters == 0) return false;
    return 2.0 * ( extra_parameters - (loglikelihood_orig - loglikelihood_merged) ) <= CHECK_PARAMETER;
};

double aic::compute_score(state_merger *merger, apta_node* left, apta_node* right){
  return 2.0 * ( extra_parameters - (loglikelihood_orig - loglikelihood_merged) );
};

double aic::split_compute_score(state_merger *, apta_node* left, apta_node* right){
    return 2.0 * ( (loglikelihood_orig - loglikelihood_merged) - extra_parameters);
};
