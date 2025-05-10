/**
 * @file ds_initializer_pre_generate_sequences.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-03-30
 * 
 * @copyright Copyright (c) 2025
 * 
 */
 
#ifndef _DS_INITIALIZER_GENERATE_SEQUENCES_H_
#define _DS_INITIALIZER_GENERATE_SEQUENCES_H_

#include "ds_initializer_base.h"

#include <vector>

/**
 * @brief This initializer generates all possible sequences from the alphabet. Take the alphabet-size, and the long-term-dependency window, 
 * then do all of them.
 */
class ds_initializer_pre_generate_sequences : public ds_initializer_base {
  private:
    inline void recurse_layer( /*out*/ std::vector<int>& current_vector, const int depth, const int max_depth, const std::vector<int>& alphabet);  
    std::shared_ptr<distinguishing_sequences_handler> ii_handler;
  
  public:
    void init(std::shared_ptr<distinguishing_sequences_handler> ii_handler, std::unique_ptr<apta>& aut) override;
};

#endif