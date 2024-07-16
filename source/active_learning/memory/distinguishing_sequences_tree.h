/**
 * @file distinguishing_sequences_tree.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-07-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _DISTINGUISHING_SEQUENCES_TREE_H_
#define _DISTINGUISHING_SEQUENCES_TREE_H_

#include <stdexcept>
#include <unordered_map>

namespace distinguishing_sequences {
  class ds_node {

  };
};

class distinguishing_sequences_tree {

  private:
    distinguishing_sequences::ds_node root;

  public:
    distinguishing_sequences_tree(){
      throw std::exception(); // not implemented yet
    }
};

#endif