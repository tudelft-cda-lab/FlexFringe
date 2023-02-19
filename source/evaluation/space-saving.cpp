/**
 * @file space-saving.cpp
 * @author Robert Baumgartner
 * @brief This file implements Balle et al.'s merge heuristic (Bootstrapping and learning PDFA from data streams, 2012)
 * @version 0.1
 * @date 2020-05-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "state_merger.h"
#include "evaluate.h"
#include "evaluation_factory.h"
#include "parameters.h"
#include "space-saving.h"

#include <stdlib.h>
#include <math.h>
#include <vector>
#include <map>
#include <stdio.h>
#include <string>
#include<cmath>
#include <algorithm>
#include <tuple>

REGISTER_DEF_TYPE(space_saving);
REGISTER_DEF_DATATYPE(space_saving_data);

const bool USE_HOEFFDING_BOUND = false;

/**
 * @brief Construct a new space_saving data::space_saving data object
 * 
 */
//space_saving_data::space_saving_data() : alergia_data::alergia_data(), main_sketch(static_cast<int>(ceil(64./MU))) {
space_saving_data::space_saving_data() : alergia_data::alergia_data(), main_sketch(K) {
    // TODO: parameters
    //const int K = static_cast<int>(ceil(64./MU));
    for(int i=0; i<R; ++i){
        bootstrapped_sketches.push_back(PrefSpSvSketch(K));
    }
};

/**
 * @brief Adds tail and updates count-min-bootstrapped_sketches.
 * 
 * @param t The tail to add.
 */
void space_saving_data::add_tail(tail* t){
    alergia_data::add_tail(t);
    main_sketch.store(t);
    for(int i=0; i<R; ++i){
        const int index = rand() % R;
        bootstrapped_sketches[index].store(t);
    }
}

/**
 * @brief Check wether the sketches satisfy the lower bound (Proposition 3 from paper).
 * 
 * @param other The other data object.
 * @return true States do not satisfy lower bound, i.e. are dissimilar with probability=(1-DELTA). 
 * @return false Dissimilarity can not be shown.
 */
bool space_saving_data::check_lower_bound(space_saving_data* other){
    // TODO: delete
/*     if(this->node->get_number()==5 && other->node->get_number()==6){
        cout << "Upper bound test: " << this->main_sketch.test_on_lower_bound(other->main_sketch, DELTA) << endl;
    }
    if(this->node->get_number()==11 && other->node->get_number()==10){
        cout << "Here we are" << endl;
    } */
    return this->main_sketch.test_on_lower_bound(other->main_sketch, DELTA/2);
}

bool space_saving_data::hoeffding_check(space_saving_data* other) const {
    return this->main_sketch.hoeffding(other->main_sketch);
}

double space_saving_data::cosine_sim(space_saving_data* other) const {
    return this->main_sketch.cosine_similarity(other->main_sketch);
}

/**
 * @brief Get the upper bound, holds with probability=(1-DELTA) (Corollary 5 from paper).
 * 
 * @param other 
 * @return double 
 */
double space_saving_data::get_upper_bound(space_saving_data* other){
    if(R == 0){
        // we do not perform bootstrap. A warning for the user should've been thrown by now. 
        return true;
    }

    vector< tuple<double, int, int> > similarities;
    for(int i=0; i<R; ++i){
        for(int j=0; j<R; ++j){
            const auto mu_hat = this->bootstrapped_sketches[i].compute_prefix_distance(other->bootstrapped_sketches[j]);
            similarities.push_back( make_tuple(mu_hat, i, j) );
        }
    }

    vector<double> bounds;
    std::sort(similarities.begin(), similarities.end());//, compare_similarity_tuples); Note: we want those in ascending order for the tests
    for(int k=0; k<similarities.size(); ++k){
        const auto& sim_tuple = similarities[k];

        double mu_hat = get<0>(sim_tuple);
        int left_index = get<1>(sim_tuple);
        int right_index = get<2>(sim_tuple);
        
        const auto estimate = this->bootstrapped_sketches[left_index].get_upper_bound_estimate(other->bootstrapped_sketches[right_index], mu_hat, DELTA/2, R, double(k+1));
        bounds.push_back(estimate);
    }

    if(bounds.size() == 0){
        throw new runtime_error("Runtime-error: The size of the bounds object should not be zero.");
    }

    double min_value = bounds[0];
    for(int i=1; i<bounds.size(); ++i){
        if(bounds[i] < min_value){
            min_value = bounds[i];
        }
    }

    return min_value;
}


/**
 * @brief Updates the data of a node when merged during the state-merging process.
 * 
 * @param right The evaluation data of the node.
 */
void space_saving_data::update(evaluation_data* right) {
    alergia_data::update(right);
    space_saving_data* other = dynamic_cast<space_saving_data*>(right);

    this->main_sketch + other->main_sketch;
    for(int i=0; i<R; ++i){
        bootstrapped_sketches[i] + other->bootstrapped_sketches[i];
    }
};


/**
 * @brief Undo the changes in the data of a node when a merge is canceled.
 * 
 * @param right The data of the node.
 */
void space_saving_data::undo(evaluation_data *right) {
    alergia_data::undo(right);
    const space_saving_data* other = dynamic_cast<space_saving_data*>(right);

    throw runtime_error("Undo not possible with space saving heuristic.");

/*     for(int j = 0; j < NSTEPS_SKETCHES; j++){
        this->bootstrapped_sketches[j] - other->bootstrapped_sketches[j];
    } */
};

void space_saving_data::print_state_label(iostream& output){
    alergia_data::print_state_label(output);
    const int NUM_EXAMPLES = 10;

    const auto f_list = this->main_sketch.get_most_frequent(NUM_EXAMPLES);
    output << "\n\nSize: " << this->main_sketch.get_size();
    output << "\nFrequencies:\n";
    for(auto f: f_list){
        output << "S: " << f.first << ", f: " << f.second << "\n";
    }
};

/**
 * @brief Merge consistent?
 * 
 * @param merger The state merger.
 * @param left The left hand node.
 * @param right The right hand node.
 * @return true The merge is consistent. 
 * @return false Merge is insonsistent.
 */
bool space_saving::consistent(state_merger *merger, apta_node *left, apta_node *right) {
    if (inconsistency_found) return false;
    if(left->get_size() <= STATE_COUNT || right->get_size() <= STATE_COUNT) return true;

    space_saving_data* left_data = dynamic_cast<space_saving_data*>(left->get_data());
    space_saving_data* right_data = dynamic_cast<space_saving_data*>(right->get_data());

    if(USE_HOEFFDING_BOUND){
        if(!left_data->hoeffding_check(right_data)){
            inconsistency_found = true; 
            return false; 
        }
        return true; 
    }

    // --------------------------------- old work ---------------------------------
    if(left_data->check_lower_bound(right_data)){
        // lower bound not satisfied, i.e. probably dissimilar 
        inconsistency_found = true;
        return false;
    }
    
    const double upper_bound = left_data->get_upper_bound(right_data);

    if(upper_bound <= MU){
        score = MU - upper_bound; // MU-upper_bound;
        return true;
    }

    inconsistency_found = true;
    return false;
};

/**
 * @brief Computes the score of a merge.
 * 
 * @param merger The state merger.
 * @param left The left hand node.
 * @param right The right hand node.
 * @return double The merge score.
 */
double space_saving::compute_score(state_merger *merger, apta_node *left, apta_node *right) {
    if(USE_HOEFFDING_BOUND){
        space_saving_data* left_data = dynamic_cast<space_saving_data*>(left->get_data());
        space_saving_data* right_data = dynamic_cast<space_saving_data*>(right->get_data());
        score = left_data->cosine_sim(right_data);
    }

    return score;
}

void space_saving::initialize_before_adding_traces() {
    alergia::initialize_before_adding_traces();

    if(MERGE_WHEN_TESTING){
        MERGE_WHEN_TESTING = false;
        cout << "WARNING: testmerge option should not be turned on with this heuristic and has been turned off automatically. The nature of the sketches prohibits undo-operations." << endl;
    }

    if(EPSILON <= double(0)){
       throw new runtime_error("Epsilon parameter must be larger than 0.");
    }

    if(R == 0){
        cout << "WARNING: R parameter chosen to be zero. No bootstrap will be performed, and this can result in bad and/or wrong results." << endl;
    }
}

/**
 * @brief Reset everything.
 * 
 * @param merger The state merger.
 */
void space_saving::reset(state_merger *merger) {
    alergia::reset(merger);
    inconsistency_found = false;

    score = 0;
};