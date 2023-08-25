/**
 * @file search_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-04-13
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _AL_SEARCH_BASE_H_
#define _AL_SEARCH_BASE_H_

#include "sul_base.h"
#include "source/input/inputdata.h"

#include <list>
#include <optional>
#include <memory>

class search_base {
  protected: 
    const int MAX_SEARCH_DEPTH;
  public:
    search_base(const int max_depth) : MAX_SEARCH_DEPTH(max_depth){};

    virtual std::optional< std::list<int> > next(const inputdata& id) = 0;
    virtual void reset() = 0;
};

#endif
