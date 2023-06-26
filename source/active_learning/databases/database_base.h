/**
 * @file database_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-06-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef DATABASE_BASE_H_
#define DATABASE_BASE_H_

#include "apta.h"

#include <fstream>

class database_base {
  protected:
    virtual void initialize() = 0;
  public:
    database_base(){
      initialize();
    }

    virtual bool is_member(const std::list<int>& query_trace) const = 0;
    virtual bool get_suffixes_with_counts(trace* prefix) = 0;
    virtual void update_state_with_statistics(apta_node* n) = 0;
};

#endif