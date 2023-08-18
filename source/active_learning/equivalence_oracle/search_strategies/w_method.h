/**
 * @file w_method.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-04-13
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _AL_W_METHOD_SEARCH_H_
#define _AL_W_METHOD_SEARCH_H_

#include "search_base.h"

class w_method : public search_base {
  public:
    w_method(const int depth) : search_base(), MAX_SEARCH_DEPTH(depth) {};

    virtual std::optional< std::list<int> > next(const inputdata& id) override;
    std::optional< std::list<int> > next(const inputdata& id, const int lower_bound);
};

#endif
