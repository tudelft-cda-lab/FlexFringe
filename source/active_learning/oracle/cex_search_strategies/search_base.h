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

#include "source/input/inputdata.h"
#include "sul_base.h"
#include "state_merger.h"
#include "parameters.h"

#include <memory>
#include <optional>
#include <vector>

class search_base {
  protected:
    const int MAX_SEARCH_DEPTH;

  public:
    search_base() : MAX_SEARCH_DEPTH(AL_MAX_SEARCH_DEPTH){
      if(AL_MAX_SEARCH_DEPTH < 1)
        throw std::invalid_argument("invalid input: maximum search depth must be greater than 1");
    };

    virtual std::optional<std::vector<int>> next(const inputdata& id) = 0;
    virtual void reset(){};

    virtual void initialize(state_merger* merger){
    }
};

#endif
