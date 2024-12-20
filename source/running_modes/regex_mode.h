/**
 * @file regex_mode.h
 * @author Hielke Walinga
 * @brief 
 * @version 0.1
 * @date 2024-12-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _REGEX_MODE_H_
#define _REGEX_MODE_H_

#include "running_mode_base.h"

class regex_mode : public running_mode_base {
  public:
    void initialize() override;
    int run() override;
};

#endif // _REGEX_MODE_H_