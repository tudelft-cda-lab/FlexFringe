/**
 * @file classification_tree.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This class serves as the basis for the adaptive distinguishing sequences, but can be used somewhere else as well, 
 * thus the modular implementation. 
 * @version 0.1
 * @date 2024-07-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _CLASSIFICATION_TREE_H_
#define _CLASSIFICATION_TREE_H_

#include <stdexcept>
#include <unordered_map>


class classification_tree {

  private:
    struct cl_tree_node {
      public:

    };

    cl_tree_node root;

  public:
    classification_tree(){
      throw std::exception(); // not implemented yet
    }
};

#endif