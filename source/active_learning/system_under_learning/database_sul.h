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

//#include "database_headers.h" // TODO: I don't get why it does not find this one on my machine
#include "source/active_learning/databases/database_headers.h"
#include "sul_base.h"

#include <memory>

class database_sul : public sul_base {
  private:
    std::unique_ptr<database_base> database;

  public:
    database_sul() { this->database = std::make_unique<prefix_tree_database>(); };
    const sul_response do_query(const std::vector<int>& query_trace, inputdata& id) const override;

    void pre(inputdata& id) override;
    void reset() override{};

    void update_state_with_statistics(apta_node* n) { this->database->update_state_with_statistics(n); }
};

#endif