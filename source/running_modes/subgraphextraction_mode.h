/**
 * @file subgraphextraction_mode.h
 * @author Sicco Verwer (s.e.verwer@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _SUBGRAPH_MODE_H
#define _SUBGRAPH_MODE_H

#include "running_mode_base.h"

class subgraphextraction_mode : public running_mode_base {
  public:
    void initialize() override;
    int run() override;
};


#endif // _SUBGRAPH_MODE_H
