/**
 * @file prefix_tree_database.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief This database simply uses an AUT to store input traces. It can then be further used by the 
 * database-SUT to obtain statistics and memberships.
 *  
 * @version 0.1
 * @date 2023-06-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef PREFIX_DATABASE
#define PREFIX_DATABASE

#include "database_base.h"
#include "apta.h"

#include <unordered_map>
#include <fstream>
#include <memory>

class prefix_tree_database : public database_base {
  protected:

    std::unique_ptr<apta> the_tree;

    virtual void initialize();
  public:
    prefix_tree_database(){
      database_base::database_base();
    }
};

#endif