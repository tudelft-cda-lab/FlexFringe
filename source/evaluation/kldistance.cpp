#include <stdlib.h>
#include <math.h>
#include <vector>
#include <map>
#include <stdio.h>

#include "state_merger.h"
#include "evaluate.h"
#include "kldistance.h"

#include "parameters.h"

REGISTER_DEF_DATATYPE(kl_data);
REGISTER_DEF_TYPE(kldistance);

void kl_data::update(evaluation_data* right){
    alergia_data::update(right);
    kl_data* other = (kl_data*)right;
        for(prob_map::iterator it = other->original_probability_count.begin(); it != other->original_probability_count.end(); ++it){
            original_probability_count[it->first] = opc(it->first) + it->second;
        }
    if(FINAL_PROBABILITIES)
        original_finprob_count = fpc() + other->fpc();
};

void kl_data::undo(evaluation_data* right){
    alergia_data::undo(right);
    kl_data* other = (kl_data*)right;
    for(prob_map::iterator it = other->original_probability_count.begin(); it != other->original_probability_count.end(); ++it){
        original_probability_count[it->first] = opc(it->first) - it->second;
    }
    if(FINAL_PROBABILITIES)
        original_finprob_count = fpc() - other->fpc();
};

bool kldistance::consistent(state_merger *merger, apta_node* left, apta_node* right, int depth){
    return count_driven::consistent(merger, left, right, depth);
};

void kldistance::update_perplexity(apta_node* left, float op_count_left, float op_count_right, float count_left, float count_right, float left_divider, float right_divider){
    if(true || already_merged(left) == false){
        if(count_left != 0){
            perplexity += op_count_left * log(count_left / left_divider);
            perplexity -= op_count_left * log((count_left + count_right) / (left_divider + right_divider));
        }
        if(count_right != 0){
            perplexity += op_count_right * log(count_right / right_divider);
            perplexity -= op_count_right * log((count_left + count_right) / (left_divider + right_divider));
        }
    } else {
        if(count_left != 0){
            perplexity += op_count_left * log(count_left / left_divider);
            perplexity -= op_count_left * log((count_left + count_right) / (left_divider + right_divider));
        }
        if(count_right != 0){
            perplexity += op_count_right * log(count_right / right_divider);
            perplexity -= op_count_right * log((count_left + count_right) / (left_divider + right_divider));
        }
    }
    if(count_left > 0.0 && count_right > 0.0) extra_parameters = extra_parameters + 1;
};

/* Kullback-Leibler divergence (KL), MDI-like, computes the KL value/extra parameters and uses it as score and consistency */
void kldistance::update_score(state_merger *merger, apta_node* left, apta_node* right){
    evaluation_function::update_score(merger, left, right);
    kl_data* l = (kl_data*) left->get_data();
    kl_data* r = (kl_data*) right->get_data();

    float left_divider = (float)l->pos_paths() + l->neg_paths();
    float right_divider = (float)r->pos_paths() + r->neg_paths();
    if(FINAL_PROBABILITIES){
        left_divider += (float)l->pos_final() + l->neg_final();
        right_divider += (float)r->pos_final() + r->neg_final();
    }

    if(left_divider < STATE_COUNT || right_divider < STATE_COUNT) return;
    if(left_divider < 1 || right_divider < 1) return;

        for(num_map::iterator it = l->counts_begin(); it != l->counts_end(); ++it){
            int symbol = it->first;
            int left_count = it->second;
            int right_count = r->count(symbol);
            update_perplexity(left, l->opc(symbol), r->opc(symbol), left_count, right_count, left_divider, right_divider);
        }        

    if(FINAL_PROBABILITIES){
        int left_count = l->pos_final() + l->neg_final();
        int right_count = r->pos_final() + r->neg_final();
        update_perplexity(left, l->fpc(), r->fpc(), left_count, right_count, left_divider, right_divider);
    }
};

bool kldistance::compute_consistency(state_merger *merger, apta_node* left, apta_node* right){
  if (inconsistency_found) return false;
  if (extra_parameters == 0) return false;

  if ((perplexity / (float)extra_parameters) > CHECK_PARAMETER) return false;

  return true;
};

double kldistance::compute_score(state_merger *merger, apta_node* left, apta_node* right){
  if (inconsistency_found == true) return false;
  if (extra_parameters == 0) return -1;

  double val = (perplexity / (double)extra_parameters);
  
  return 100000 - (int)(val * 100.0);
};

void kldistance::reset(state_merger *merger){
  alergia::reset(merger);
  inconsistency_found = false;
  perplexity = 0;
  extra_parameters = 0;
};

void kldistance::initialize_after_adding_traces(state_merger* merger){
    for(merged_APTA_iterator Ait = merged_APTA_iterator(merger->get_aut()->get_root()); *Ait != 0; ++Ait) {
        apta_node *node = *Ait;
        kl_data *l = (kl_data *) node->get_data();

        float divider = l->pos_paths() + l->neg_paths();
        if(FINAL_PROBABILITIES) divider += l->pos_final() + l->neg_final();

        //float probability_mass = (float)node->get_size() / (float)merger->get_dat()->get_num_sequences();

        if(divider < 1) continue;

            for(num_map::iterator it = l->counts_begin(); it != l->counts_end(); ++it){
                int symbol = it->first;
                float count = it->second;
                l->original_probability_count[symbol] = count * (count / divider);
            }

        if(FINAL_PROBABILITIES){
            float count = l->pos_final() + l->neg_final();
            l->original_finprob_count = count *  (count / divider);
        }
    }
};

