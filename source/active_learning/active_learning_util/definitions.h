/**
 * @file definitions.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Basic definitions most active learning algorithms will use.
 * @version 0.1
 * @date 2023-02-21
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _ACTIVE_LEARNING_DEFINITIONS_H_
#define _ACTIVE_LEARNING_DEFINITIONS_H_

#include <vector>
#include <map>
#include <string>

namespace active_learning_namespace{
  const int EPS = -1; // empty symbol special character. flexfringe does not map to -1 by design.
  
  typedef std::vector<int> pref_suf_t;
}

#endif