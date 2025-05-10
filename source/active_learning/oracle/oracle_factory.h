/**
 * @file oracle_factory.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-05-10
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef _ORACLE_FACTORY_H_
#define _ORACLE_FACTORY_H_

#include "base_oracle.h"
#include "sul_base.h"

#include <string_view>
#include <memory>

class oracle_factory {
  public:
    oracle_factory() = delete;

    static std::unique_ptr<base_oracle> create_oracle(const std::shared_ptr<sul_base>& sul, std::string_view oracle_name, const std::shared_ptr<distinguishing_sequences_base>& ii_handler);
};

#endif