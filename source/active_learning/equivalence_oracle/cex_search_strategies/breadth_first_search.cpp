/**
 * @file breadth_first_search.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief
 * @version 0.1
 * @date 2023-04-13
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "breadth_first_search.h"

#include <cassert>
#include <iostream>
#include <list>
#include <utility>

using namespace std;


/**
 * @brief Maps the vector of indices into a vector that the equivalence query can actually work with. 
 * 
 * @param alphabet The alphabet.
 * @param redo A minor optimization parameter. If true then redo the whole vector, else we only change the last element.
 * Can be optimized further for other indices, but no need to. It is easy to read like this.
 * @return std::vector<int> The mapped vector.
 */
std::vector<int> bfs_strategy::get_vector_from_indices(const vector<int>& alphabet, const bool redo) const {
    static vector<int> current_vector;

    if(redo){
        current_vector = vector<int>(vector_idxs.size());
        for(int i = 0; i<current_vector.size(); ++i){
            current_vector[i] = alphabet[vector_idxs[i]];
        }

        return current_vector;
    }

    const int last_idx = current_vector.size() - 1;
    current_vector[last_idx] = alphabet[vector_idxs[last_idx]];

    return current_vector;
}

/**
 * @brief Gets the next vector by first updating the vector_idx representation of the search, and 
 * then constructing the internal alphabet representation of that.
 */
std::vector<int> bfs_strategy::get_next_vector(const vector<int>& alphabet) {
    static const int last_alphabet_idx = alphabet.size()-1;

    if(vector_idxs[vector_idxs.size()-1] == last_alphabet_idx){
        int depth = vector_idxs.size()-1;
        while(depth >= 0 && vector_idxs[depth] == last_alphabet_idx){
            vector_idxs[depth] = 0;
            depth--;
        }
        
        if(depth < 0){
            vector_idxs.push_back(0);
        }
        else{
            const int idx = vector_idxs[depth];
            vector_idxs[depth] = alphabet[idx+1];
        }

        return get_vector_from_indices(alphabet, true);
    }
    else{
        // we increase only the last one and that's it
        const int idx = vector_idxs[vector_idxs.size()-1];
        vector_idxs[vector_idxs.size()-1] = alphabet[idx+1];
        
        return get_vector_from_indices(alphabet, false);
    }
}



/**
 * @brief Normal BFS through the alphabet, testing for size. I implemented this using two stacks. This requires us
 * lots of copying. An alternative method would be to instead implement recursively, which comes at the cost of the
 * heap, but that does not matter much if we assume the max depth to be not too extreme.
 * 
 * TODO: We can rewrite this function memory efficient by only storing indices of the current strings, rather than 
 * storing the prefixes.
 *
 * @param sul The SUl.
 * @param id The inputdata.
 * @return optional< const vector<int> > A counterexample if found.
 */
optional<vector<int>> bfs_strategy::next(const inputdata& id) {
    static const vector<int>& alphabet = id.get_alphabet();
    const static int alph_size = alphabet.size();
    
    if (vector_idxs.size() == 0) [[unlikely]] {
        vector_idxs.push_back(0);
        return get_vector_from_indices(alphabet, true);
    }
    else if(vector_idxs.size() == MAX_SEARCH_DEPTH) [[unlikely]] {
        return nullopt;
    }

    return make_optional(get_next_vector(alphabet));
}
