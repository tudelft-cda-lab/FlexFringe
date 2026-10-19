/**
 * @file cms.cpp
 * @author Robert Baumgartner, Raffail Skoulos
 * @brief This file implements count-min-sketch based state merging.
 * @version 0.1
 * @date 2020-05-29
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "state_merger.h"
#include "evaluate.h"
#include "evaluation_factory.h"
#include "parameters.h"
#include "cms.h"
#include "countminsketch.h"

#include <stdlib.h>
#include <math.h>
#include <vector>
#include <map>
#include <stdio.h>
#include <string>
#include<cmath>
#include <algorithm>
#include <tuple>

REGISTER_DEF_TYPE(cms);
REGISTER_DEF_DATATYPE(cms_data);

// from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
// alternatively, use boost library. TODO: use seed, so we can use multiple hash-functions
int hash_vec(std::vector<int> const& vec){
  int res = vec.size();
  res ^= vec[0] + 0x9e3779b9 + (res << 6) + (res >> 2);
  return res;
}

int hash_vec(int seed, const int elem){
  seed ^= elem + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}

int hash_vec(std::vector<int> const& vec, const int i){
  int res = vec.size();
  res ^= vec[i] + 0x9e3779b9 + (res << 6) + (res >> 2);
  return res;
}

/**
 * @brief Construct a new cms data::cms data object
 * 
 */
cms_data::cms_data() : alergia_data::alergia_data() {
    static bool initialized = false;
    if(!initialized){
        for(int i = 0; i < NSTEPS_SKETCHES; ++i){
            // TODO: initialize hash functions for containers with random seeds
        }
        initialized = true;
    }

    for(int i = 0; i < NSTEPS_SKETCHES; ++i){
        sketches.push_back( CountMinSketch<int>(NROWS_SKETCHES, NCOLUMNS_SKETCHES + 1) );
    }
};


/**
 * @brief Get all the n_grams.
 * 
 * @param tails A vector with the tails.
 * @return const vector<int> 
 */
const std::vector<int> cms_data::get_n_grams(tail* t, const int n_steps) const {
    // TODO: this approach can't handle negative numbers yet
    assert(n_steps > 0);
    std::vector<int> res;

    for(int i = 0; i < n_steps; ++i){
        if(t == 0){
            i = n_steps; // kill the inner loop
        }

        const int symbol = t->get_symbol();
        if(symbol == -1){ // termination symbol via flexfringe-convention
            res.push_back(-1);
            i = n_steps; // kill the inner loop
        }
        else{
            res.push_back(symbol);
        }
        t = t->future();
    }
    return res;
}


/**
 * @brief Adds tail and updates count-min-sketches.
 * 
 * @param t The tail to add.
 */
void cms_data::add_tail(tail* t){
    alergia_data::add_tail(t);
    //evaluation_data::add_tail(t);

    if(t->future() != 0 && t->future()->future() == 0){ // this sequence terminates here
        sketches[0].storeAt(NCOLUMNS_SKETCHES, 0); // extra row for termination symbol
    }
    else{
        auto short_term_n_grams = get_n_grams(t, NSTEPS_SKETCHES);
        int res = 0;
        int hashval = hash_vec(short_term_n_grams, 0);
        sketches[0].store(hashval, 0, NCOLUMNS_SKETCHES);
        for(int j = 1; j < NSTEPS_SKETCHES; ++j){
            if(short_term_n_grams[j] == -1){
                sketches[j].storeAt(NCOLUMNS_SKETCHES, 0);
            }
            else{
                hashval = hash_vec(short_term_n_grams, j);
                sketches[j].store(hashval, 0, NCOLUMNS_SKETCHES);
            }
        }
    }
}


/**
 * @brief Updates the data of a node when merged during the state-merging process.
 * 
 * @param right The evaluation data of the node.
 */
void cms_data::update(evaluation_data* right) {
    alergia_data::update(right);
    const cms_data* other = dynamic_cast<cms_data*>(right);

    for(int j = 0; j < NSTEPS_SKETCHES; j++){
        this->sketches[j] + other->sketches[j];
    }
};


/**
 * @brief Undo the changes in the data of a node when a merge is canceled.
 * 
 * @param right The data of the node.
 */
void cms_data::undo(evaluation_data *right) {
    alergia_data::undo(right);
    const cms_data* other = dynamic_cast<cms_data*>(right);

    for(int j = 0; j < NSTEPS_SKETCHES; j++){
        this->sketches[j] - other->sketches[j];
    }
};

void cms_data::print_state_label(std::iostream& output){
    alergia_data::print_state_label(output);

/*     output << "\n\nShort term:\n";
    for(auto sketch: short_terms){
        const auto& row = sketch.getSketch();
        for(auto symbol: row[0]){
            output << symbol << " ";
        }
        output << "\n";
    } */
};

/**
 * @brief Checks if a merge is consistent by comparing the KL divergence of the 3 cms
 * structures with a user-defined threshold.
 * 
 * @param merger The state merger.
 * @param left The left hand node.
 * @param right The right hand node.
 * @return true The merge is consistent. 
 * @return false Merge is insonsistent.
 */
bool cms::consistent(state_merger *merger, apta_node *left, apta_node *right) {
    if (inconsistency_found) return false;
    if(left->get_size() <= STATE_COUNT || right->get_size() <= STATE_COUNT) return true;

    const cms_data* left_data = dynamic_cast<cms_data*>(left->get_data());
    const cms_data* right_data = dynamic_cast<cms_data*>(right->get_data());

    int thresh = 20;
    for(int j = 1; j < NSTEPS_SKETCHES; j++){
        if(left_data->get_sketches()[j].getZeroSize() < thresh && right_data->get_sketches()[j].getSize() < thresh){
            inconsistency_found = true;
            return false;
        }
    }
    
    bool all_sketches_similar = true;
    if(DISTANCE_METRIC_SKETCHES == 1){
        for(int j = 0; j < NSTEPS_SKETCHES; j++){
            //if(j>1) break;
            all_sketches_similar = all_sketches_similar && CountMinSketch<int>::hoeffding(left_data->get_sketches()[j], right_data->get_sketches()[j], CHECK_PARAMETER);
            if(!all_sketches_similar) break;
        }
    }
    else if(DISTANCE_METRIC_SKETCHES == 2){

        for(int j = 0; j < NSTEPS_SKETCHES; j++){
            all_sketches_similar = all_sketches_similar && CountMinSketch<int>::hoeffdingWithPooling(left_data->get_sketches()[j], right_data->get_sketches()[j], CHECK_PARAMETER, SYMBOL_COUNT, STATE_COUNT, CORRECTION);
            if(!all_sketches_similar) break;
        }
    }
    else if(DISTANCE_METRIC_SKETCHES == 3){
        for(int j = 0; j < NSTEPS_SKETCHES; j++){
            float score = CountMinSketch<int>::hoeffdingScore(left_data->get_sketches()[j], right_data->get_sketches()[j], CHECK_PARAMETER);
            if(score < 0){
                all_sketches_similar = false;
                break;
            }
            scoresum += score;
        }
    }
    else{
        throw std::invalid_argument("Input parameter cms-distance metric should only be a single digit from 1 to 7.");
    }

    if (!all_sketches_similar){
        inconsistency_found = true;
        return false;
    }

    return true;
};

/**
 * @brief Computes the score of a merge.
 * 
 * @param merger The state merger.
 * @param left The left hand node.
 * @param right The right hand node.
 * @return double The merge score.
 */
double cms::compute_score(state_merger *merger, apta_node *left, apta_node *right) {
    const cms_data* left_data = dynamic_cast<cms_data*>(left->get_data());
    const cms_data* right_data = dynamic_cast<cms_data*>(right->get_data());

    if(DISTANCE_METRIC_SKETCHES == 1){
        scoresum = 0; 
        const float max_size = static_cast<float>(left_data->get_sketches()[0].getZeroSize() + right_data->get_sketches()[0].getZeroSize());
        for(int j = 0; j < NSTEPS_SKETCHES; j++){
            float current_size = static_cast<float>(left_data->get_sketches()[j].getZeroSize() + right_data->get_sketches()[j].getZeroSize());
            scoresum += CountMinSketch<int>::cosineSimilarity(left_data->get_sketches()[j], right_data->get_sketches()[j]);
        }
    }
    else if(DISTANCE_METRIC_SKETCHES == 2){
        scoresum = 0; 
        const float max_size = static_cast<float>(left_data->get_sketches()[0].getZeroSize() + right_data->get_sketches()[0].getZeroSize());
        for(int j = 0; j < NSTEPS_SKETCHES; j++){
            float current_size = static_cast<float>(left_data->get_sketches()[j].getZeroSize() + right_data->get_sketches()[j].getZeroSize());
            scoresum += (current_size / max_size) * CountMinSketch<int>::cosineSimilarity(left_data->get_sketches()[j], right_data->get_sketches()[j]);
        }
    }
    else if(DISTANCE_METRIC_SKETCHES == 3){
        return scoresum;        
    }

    return static_cast<double>(scoresum);
}

/* void cms::initialize(state_merger* merger) {
    alergia::initialize(merger);
} */

/**
 * @brief Reset everything.
 * 
 * @param merger The state merger.
 */
void cms::reset(state_merger *merger) {
    alergia::reset(merger);
    inconsistency_found = false;

    scoresum = 0;
};