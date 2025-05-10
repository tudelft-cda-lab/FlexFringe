/**
 * @file ds_initializers.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-04-06
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <string>
#include <unordered_map>
#include <string_view>

#ifndef _DS_INITIALIZER_REGISTRATION_H_
#define _DS_INITIALIZER_REGISTRATION_H_

class ds_initializer_registration final {
  public:
    enum class ds_initializers_t{
      collect_from_apta = 0,
      pre_generate_sequences = 1
    };

    static std::string_view get_initializer_name(const ds_initializers_t init_instance);
};

#endif // _DS_INITIALIZER_REGISTRATION_H_