/**
 * @file dfa_conflict_search_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Base class for all conflict search algorithms that work 
 * on a DFA.
 * @version 0.1
 * @date 2024-07-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _AL_DFA_CONFLICT_SEARCH_BASE_H_
#define _AL_DFA_CONFLICT_SEARCH_BASE_H_

#include "conflict_search_base.h"
#include "common_functions.h"

class dfa_conflict_search_base : public conflict_search_base {
  protected:
    virtual int get_teacher_response(const std::vector<int>& cex, const std::unique_ptr<base_teacher>& teacher, inputdata& id) const;

    int parse_dfa(const vector<int>& seq, apta& hypothesis, inputdata& id);

  public:
    dfa_conflict_search_base(){};

    std::pair< std::vector<int>, std::optional<response_wrapper> > get_conflict_string(const std::vector<int>& cex, apta& hypothesis, const std::unique_ptr<base_teacher>& teacher, inputdata& id) = 0;
};

#endif
