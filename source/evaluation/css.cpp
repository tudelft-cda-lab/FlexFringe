/**
 * @file css.cpp
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
#include "css.h"
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

REGISTER_DEF_TYPE(css);
REGISTER_DEF_DATATYPE(css_data);

/**
 * @brief Construct a new css data::css data object
 * 
 */
css_data::css_data() : alergia_data::alergia_data() {
    static bool initialized = false;
    static bool symbols_initialized = false;

    if(!initialized && CONDITIONAL_PROB && MINHASH){
        // initialize the minhash functions
        vector<int> alphabet;
        for(int i = 0; i < ALPHABET_SIZE; ++i){
            alphabet.push_back(i);
        }
    
        for(int i = 0; i < MINHASH_SIZE; ++i){
            auto rng = std::default_random_engine {};
            shuffle(alphabet.begin(), alphabet.end(), rng);
            minhash_functions.push_back( minhash_func(alphabet) );
        }
        initialized = true;
    }

    if(!symbols_initialized){
        for(int i = 0; i < NSTEPS_SKETCHES; ++i){
            seen_symbols.push_back( set<int>() );
        }
        symbols_initialized = true;
    }

    for(int i = 0; i < NSTEPS_SKETCHES; ++i){
        sketches.push_back( CountMinSketch(NROWS_SKETCHES, NCOLUMNS_SKETCHES) );
    }
};

/**
 * @brief Minhash a single set.
 * 
 * @param shingle_set The set to hash.
 * @param n_gram_size The size of the original n-gram. Needed to determine to what dimension to reduce the shingle-set.
 * @return const vector<int> Result as a new, hashed "n-gram".
 */
const vector<int> css_data::minhash_set(const set<int>  shingle_set, const int n_gram_size) const{
    if(shingle_set.count(-1) > 0){
        // this was the termination character
        return vector<int>{-1};
    }

    vector<int> res;
    const int NUMBER_OF_HASHES = min(n_gram_size, MINHASH_SIZE);

    for(int i = 0; i < NUMBER_OF_HASHES; ++i){
        const auto& current_hash_function = minhash_functions[i]; 
        res.push_back(current_hash_function.get_mapping(shingle_set));
    }

    return res;
}

/**
 * @brief Apply the min-hash scheme to all the sets given. In this implementation, we hash down to a size of 2, or 1 if coming from a 1-gram.
 * 
 * @param shingle_sets A sorted list of ngrams transformed to sets. Assumption: They are sorted, i.e. the one at position 0 is the 1-gram, position 1 the 2-gram, and so on.
 * @return vector<int> Vector of hashed and encoded shingle-sets.
 */
const vector<int> css_data::encode_sets(const vector< set<int> > shingle_sets) const {
    vector<int> res;

    vector< vector<int> > hashed_sets;
    for(int i = 0; i < shingle_sets.size(); ++i){
        const set<int> shingle_set = shingle_sets[i];
        hashed_sets.push_back(minhash_set(shingle_set, i+1));
    }

    for(const auto& hashed_set: hashed_sets){
        if(hashed_set.at(0) == -1){
            // the termination character
            res.push_back(-1);
            break;
        }

        int code = 0;
        static const double SPACE_SIZE = pow(ALPHABET_SIZE, MINHASH_SIZE);
        double space_size = SPACE_SIZE;

        for(int k = 0; k<hashed_set.size(); ++k){
            code = code + static_cast<double>(hashed_set.at(k) + 1) * (space_size / MINHASH_SIZE); // we do +1, because zero is reserved for "shorter than this ngram". This is our "zero-padding"
            space_size = space_size / MINHASH_SIZE;
        }
        res.push_back(code);
    }

    return res;
}

/**
 * @brief Get all the n_grams.
 * 
 * @param tails A vector with the tails.
 * @return const vector<int> 
 */
const vector<int> css_data::get_symbols_as_list(tail* t) const {
    assert(NSTEPS_SKETCHES > 0);
    vector<int> res;

    if(!CONDITIONAL_PROB){
        for(int i = 0; i < NSTEPS_SKETCHES; ++i){
            if(t == 0){
                i = NSTEPS_SKETCHES; // kill the inner loop
                break;
            }

            const int symbol = t->get_symbol();
            res.push_back(symbol);
            if(symbol == -1){ // termination symbol via flexfringe-convention
                break; // kill the inner loop
            }

            t = t->future();
        }
    }
    else if(MINHASH){
        // mapping the sequences to unique values, see e.g. Learning behavioral fingerprints from Netflows using Timed Automata (Pellegrino et al.)

        // step 1: construct the sets
        vector< set<int> > shingles;
        set<int> current_shingle;
        for(int i = 0; i < NSTEPS_SKETCHES; ++i){
            if(t == 0){
                i = NSTEPS_SKETCHES; // kill the inner loop
                break;
            }

            const int symbol = t->get_symbol();
            if(symbol == -1){ // termination symbol via flexfringe-convention
                shingles.push_back(set<int>{-1}); // code cannot be negative, hence no clash possible 
                break;
            }
            // map the symbols to integer values from 1 to ALPHABET_SIZE
            if(symbol_to_mapping.count(symbol) == 0) symbol_to_mapping[symbol] = symbol_to_mapping.size();

            const int feature_mapping = static_cast<int>(symbol_to_mapping.at(symbol));
            current_shingle.insert(feature_mapping);
            shingles.push_back(set<int>(current_shingle));

            t = t->future();
        }
        
        // 2. encode the sets
        res = encode_sets(shingles);
    }
    else{
        // mapping the sequences to unique values, see e.g. Learning behavioral fingerprints from Netflows using Timed Automata (Pellegrino et al.)
        int code = 0;
        static const double SPACE_SIZE = pow(ALPHABET_SIZE, NSTEPS_SKETCHES);
        double space_size = SPACE_SIZE;

        for(int i = 0; i < NSTEPS_SKETCHES; ++i){
            if(t == 0){
                i = NSTEPS_SKETCHES; // kill the inner loop
                break;
            }

            const int symbol = t->get_symbol();
            if(symbol == -1){ // termination symbol via flexfringe-convention
                res.push_back(-1); // code cannot become -1
                break;
            }

            if(symbol_to_mapping.count(symbol) == 0) symbol_to_mapping[symbol] = symbol_to_mapping.size() + 1;

            const auto feature_mapping = symbol_to_mapping.at(symbol);
            code = code + static_cast<int>(symbol_to_mapping.at(symbol) * (space_size / ALPHABET_SIZE));
            res.push_back(code);
            space_size = space_size / ALPHABET_SIZE;

            t = t->future();
        }
    }

    return res;
}



/**
 * @brief Adds tail and updates count-min-sketches.
 * 
 * @param t The tail to add.
 */
void css_data::add_tail(tail* t){
    alergia_data::add_tail(t);
    //evaluation_data::add_tail(t);

    auto n_grams = get_symbols_as_list(t);
    assert(n_grams.size() <= sketches.size());

    for(int i=0; i<n_grams.size(); ++i){
        const auto symbol = n_grams[i];
        seen_symbols[i].insert(symbol);
        sketches[i].store(symbol);
    }
}


/**
 * @brief Updates the data of a node when merged during the state-merging process.
 * 
 * @param right The evaluation data of the node.
 */
void css_data::update(evaluation_data* right) {
    alergia_data::update(right);
    const css_data* other = dynamic_cast<css_data*>(right);

    for(int j = 0; j < NSTEPS_SKETCHES; j++){
        this->sketches[j] + other->sketches[j];
    }
};


/**
 * @brief Undo the changes in the data of a node when a merge is canceled.
 * 
 * @param right The data of the node.
 */
void css_data::undo(evaluation_data *right) {
    alergia_data::undo(right);
    const css_data* other = dynamic_cast<css_data*>(right);

    for(int j = 0; j < NSTEPS_SKETCHES; j++){
        this->sketches[j] - other->sketches[j];
    }
};

void css_data::print_state_label(iostream& output){
    alergia_data::print_state_label(output);

/*     output << "\n\nSketches:\n";
    for(auto sketch: this->sketches){
        for(const auto& row: sketch.getSketch()){
            for(auto symbol: row){
                output << symbol << " ";
            }
            output << "\n";
        }
        output << "final: " << sketch.getFinalCounts() << "\n";
    }
    output << "\n\n"; */
};

/**
 * @brief Checks if a merge is consistent by comparing the KL divergence of the 3 css
 * structures with a user-defined threshold.
 * 
 * @param merger The state merger.
 * @param left The left hand node.
 * @param right The right hand node.
 * @return true The merge is consistent. 
 * @return false Merge is insonsistent.
 */
bool css::consistent(state_merger *merger, apta_node *left, apta_node *right) {
    if (inconsistency_found) return false;
    if(left->get_size() <= STATE_COUNT || right->get_size() <= STATE_COUNT) return true;

    const css_data* left_data = dynamic_cast<css_data*>(left->get_data());
    const css_data* right_data = dynamic_cast<css_data*>(right->get_data());
    
    bool all_sketches_similar = true;
    for(int j = 0; j < NSTEPS_SKETCHES; j++){
        const auto& symbols = left_data->get_seen_symbols(j);

        all_sketches_similar = all_sketches_similar && CountMinSketch::hoeffding(left_data->get_sketches()[j], right_data->get_sketches()[j], symbols);
        //all_sketches_similar = all_sketches_similar && CountMinSketch::hoeffding(left_data->get_sketches()[j], right_data->get_sketches()[j]);
        if(!all_sketches_similar) break;
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
double css::compute_score(state_merger *merger, apta_node *left, apta_node *right) {
    const css_data* left_data = dynamic_cast<css_data*>(left->get_data());
    const css_data* right_data = dynamic_cast<css_data*>(right->get_data());

    scoresum = 0; 
    for(int j = 0; j < NSTEPS_SKETCHES; j++){
        const auto& symbols = left_data->get_seen_symbols(j);

        scoresum += CountMinSketch::cosineSimilarity(left_data->get_sketches()[j], right_data->get_sketches()[j], symbols);
    }

    return scoresum;
}

/**
 * @brief Reset everything.
 * 
 * @param merger The state merger.
 */
void css::reset(state_merger *merger) {
    alergia::reset(merger);
    inconsistency_found = false;

    scoresum = 0;
};