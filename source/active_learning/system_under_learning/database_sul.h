/**
 * @file database_sul.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The SUL used to query along the databases. This is possibly the most important SUL 
 * in this library.
 * @version 0.1
 * @date 2023-06-23
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _DATABASE_SUL_H_
#define _DATABASE_SUL_H_

#include "database_headers.h"
#include "sul_base.h"

#include <memory>

class dabase_sul : public sul_base {
  friend class base_teacher;
  friend class eq_oracle_base;

  private:
    std::unique_ptr<database_base> database;

  protected:
    virtual void post() override {};
    virtual void step() override {};
    virtual void reset() override {};

    virtual bool is_member(const std::list<int>& query_trace) const override;
    virtual const int query_trace(const std::list<int>& query_trace, inputdata& id) const override;
    
  public:
    dabase_sul(){
      database = std::unique_ptr<database_base>(new prefix_tree_database());
    };
    virtual void pre(inputdata& id) override;
};

#endif