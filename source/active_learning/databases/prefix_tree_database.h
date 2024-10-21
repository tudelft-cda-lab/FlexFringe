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

#ifndef _PREFIX_TREE_DATABASE_H_
#define _PREFIX_TREE_DATABASE_H_

#include "apta.h"
#include "database_base.h"

#include <fstream>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

class prefix_tree_database : public database_base {
  protected:
    std::unique_ptr<apta> the_tree;

    virtual void initialize() override;
    std::vector<std::pair<trace*, int>> extract_tails_from_tree(apta_node* start);

  public:
    prefix_tree_database() { initialize(); }

    virtual bool is_member(const std::vector<int>& query_trace) const override;
    virtual void update_state_with_statistics(apta_node* n) override;
};

#endif