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
  public:
    search_base() = default;

    virtual std::optional< const std::list<int> > next(const std::shared_ptr<sul_base>& sul, const inputdata& id) = 0;
};

#endif
