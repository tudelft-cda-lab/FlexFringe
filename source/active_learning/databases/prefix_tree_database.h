/**
 * @file prefix_tree_database.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-06-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef PREFIX_DATABASE
#define PREFIX_DATABASE

#include "database_base.h"

#include <unordered_map>
#include <fstream>

class prefix_database : public database_base {
  protected:

/*     struct prefix_tree_node{
      typedef 
      public:
        unordered_map<>
    } */

    virtual void initialize(std::ifstream& input);
  public:
    prefix_database(std::ifstream& input){
      initialize(input);
    }
};

#endif