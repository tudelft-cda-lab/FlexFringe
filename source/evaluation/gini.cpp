#include "state_merger.h"
#include "evaluate.h"
#include "gini.h"

REGISTER_DEF_DATATYPE(gini_data);
REGISTER_DEF_TYPE(gini);

void gini::split_update_score_before(state_merger*, apta_node* left, apta_node* right, tail* t){
    gini_data* l = (gini_data*) left->get_data();
    gini_data* r = (gini_data*) right->get_data();

    //double total_count = l->pos_total() + l->neg_total() + r->pos_total() + r->neg_total();
    /*
    double total_left = l->pos_total() + l->neg_total();
    double total_right = r->pos_total() + r->neg_total();
    */
    double total_left = l->pos_final() + l->neg_final();
    double total_right = r->pos_final() + r->neg_final();

    //double total_pos = l->pos_total() + r->pos_total();
    //double total_neg = r->pos_total() + l->pos_total();

    /*
    double right_pos_prob = r->pos_total() / total_right;
    double right_neg_prob = r->neg_total() / total_right;
    double left_pos_prob = l->pos_total() / total_left;
    double left_neg_prob = l->neg_total() / total_left;
    */

    double right_pos_prob = r->pos_final() / total_right;
    double right_neg_prob = r->neg_final() / total_right;
    double left_pos_prob = l->pos_final() / total_left;
    double left_neg_prob = l->neg_final() / total_left;

    if(total_right == 0)
        num_split += total_left;

    if(total_right != 0)
        split_score += total_right * (1.0 - ((right_pos_prob * right_pos_prob) + (right_neg_prob * right_neg_prob)));
    if(total_left != 0)
        split_score += total_left * (1.0 - ((left_pos_prob * left_pos_prob) + (left_neg_prob * left_neg_prob)));
}

void gini::split_update_score_after(state_merger*, apta_node* left, apta_node* right, tail* t){
    gini_data* l = (gini_data*) left->get_data();
    gini_data* r = (gini_data*) right->get_data();

    //double total_count = l->pos_total() + l->neg_total() + r->pos_total() + r->neg_total();
    /*
    double total_left = l->pos_total() + l->neg_total();
    double total_right = r->pos_total() + r->neg_total();
    */
    double total_left = l->pos_final() + l->neg_final();
    double total_right = r->pos_final() + r->neg_final();

    //double total_pos = l->pos_total() + r->pos_total();
    //double total_neg = r->pos_total() + l->pos_total();

    /*
    double right_pos_prob = r->pos_total() / total_right;
    double right_neg_prob = r->neg_total() / total_right;
    double left_pos_prob = l->pos_total() / total_left;
    double left_neg_prob = l->neg_total() / total_left;
    */

    double right_pos_prob = r->pos_final() / total_right;
    double right_neg_prob = r->neg_final() / total_right;
    double left_pos_prob = l->pos_final() / total_left;
    double left_neg_prob = l->neg_final() / total_left;

    if(total_left == 0)
        num_split -= total_right;

    if(total_right != 0)
        split_score -= total_right * (1.0 - ((right_pos_prob * right_pos_prob) + (right_neg_prob * right_neg_prob)));
    if(total_left != 0)
        split_score -= total_left * (1.0 - ((left_pos_prob * left_pos_prob) + (left_neg_prob * left_neg_prob)));
}

bool gini::split_compute_consistency(state_merger *, apta_node* left, apta_node* right){
    return true;
}

double gini::split_compute_score(state_merger *, apta_node* left, apta_node* right){
    //cerr << "split: " << split_score << " " << num_split << endl;
    if(num_split == 0) return - CHECK_PARAMETER;
    return (split_score / num_split) - (CHECK_PARAMETER);
}


/* GINI impurity based state merging, computes GINI improvement for merges and splits*/
bool gini::consistent(state_merger *merger, apta_node* left, apta_node* right){
    return true;
}

bool gini::compute_consistency(state_merger *merger, apta_node* left, apta_node* right){
    return true;
}

void gini::update_score(state_merger *merger, apta_node* left, apta_node* right){
    gini_data* l = (gini_data*) left->get_data();
    gini_data* r = (gini_data*) right->get_data();

    /*
    double total_count = l->pos_total() + l->neg_total() + r->pos_total() + r->neg_total();
    double total_left = l->pos_total() + l->neg_total();
    double total_right = r->pos_total() + r->neg_total();
     */

    double total_count = l->pos_final() + l->neg_final() + r->pos_final() + r->neg_final();
    double total_left = l->pos_final() + l->neg_final();
    double total_right = r->pos_final() + r->neg_final();

    /*
    double total_pos = l->pos_total() + r->pos_total();
    double total_neg = l->neg_total() + r->neg_total();
    */
    double total_pos = l->pos_final() + r->pos_final();
    double total_neg = l->neg_final() + r->neg_final();

    double total_pos_prob = total_pos / total_count;
    double total_neg_prob = total_neg / total_count;

    /*
    double right_pos_prob = r->pos_total() / total_right;
    double right_neg_prob = r->neg_total() / total_right;
    double left_pos_prob = l->pos_total() / total_left;
    double left_neg_prob = l->neg_total() / total_left;
    */

    double right_pos_prob = r->pos_final() / total_right;
    double right_neg_prob = r->neg_final() / total_right;
    double left_pos_prob = l->pos_final() / total_left;
    double left_neg_prob = l->neg_final() / total_left;

    if(total_count != 0)
        merge_score -= total_count * (1.0 - ((total_pos_prob * total_pos_prob) + (total_neg_prob * total_neg_prob)));
    if(total_right != 0)
        merge_score += total_right * (1.0 - ((right_pos_prob * right_pos_prob) + (right_neg_prob * right_neg_prob)));
    if(total_left != 0)
        merge_score += total_left * (1.0 - ((left_pos_prob * left_pos_prob) + (left_neg_prob * left_neg_prob)));
    if(total_count != 0)
        num_merge += total_count;
};

double gini::compute_score(state_merger *merger, apta_node* left, apta_node* right){
    //cerr << "split: " << split_score << " " << num_split << " merge: " << merge_score << " " << num_merge << endl;
    if(num_split == 0 && num_merge == 0) return -1.0;
    return (merge_score / num_merge) + (CHECK_PARAMETER);
    //return (split_score / num_split) + (merge_score / num_merge) + (CHECK_PARAMETER);
};

void gini::reset(state_merger *merger){
  inconsistency_found = false;
  num_pos = 0;
  num_neg = 0;
  merge_score = 0.0;
  split_score = 0.0;
  num_split = 0.0;
  num_merge = 0.0;
};
