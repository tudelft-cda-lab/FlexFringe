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

#ifndef _LOAD_SQLDB_MODE_H_
#define _LOAD_SQLDB_MODE_H_

#include "running_mode_base.h"

#include "misc/sqldb.h"
#include "misc/utils.h"

class load_sqldb_mode : public running_mode_base {
  public:
    void initialize() override;
    int run() override;
    void generate_output() override;
};

#endif // _LOAD_SQLDB_MODE_H_
