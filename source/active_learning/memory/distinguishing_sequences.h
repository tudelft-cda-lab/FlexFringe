/**
 * @file distinguishing_sequences.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief The distinguishing sequences, not optimized. 
 * @version 0.1
 * @date 2024-09-26
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _DISTINGUISHING_SEQUENCES_H_
#define _DISTINGUISHING_SEQUENCES_H_

#include "suffix_tree.h"

#include <vector>
#include <list>
#include <unordered_map>

#include <optional>
#include <iostream>

class distinguishing_sequences {
  private:
    suffix_tree seq_store; // maps a distinguishing sequence to a node (identified by node->get_number())

  public:
    virtual bool add_sequence(const std::list<int>& s) noexcept;
    virtual std::optional< std::vector<int> > next() {
      return seq_store.next();
    }

    virtual bool contains(const std::list<int>& s){return seq_store.contains(s);}
    virtual int size() const noexcept { return seq_store.size(); }
};

#endif