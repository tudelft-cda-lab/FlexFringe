/**
 * @file ds_intializer_registratration.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-04-06
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "ds_initializer_registration.h"

#include <stdexcept>

using namespace std;

/**
 * @brief Used to unify the ds_initializers. Will hopefully lead to
 * more robust code.
 */
string_view ds_initializer_registration::get_initializer_name(const ds_initializers_t init){
  // warning: keys must match with ds_initializers_t
  static std::unordered_map<int, std::string> initializer_name_map {
    {0, "ds_init_collect_from_apta"},
    {1, "ds_init_pre_generate_sequences"}
  };

  string_view res;
  try{
    res = initializer_name_map.at(static_cast<int>(init));
  }
  catch(...){
    throw invalid_argument("Init name seems wrong, or is not implemented in ds_initializer_registration. Please check");
  }
  return res;
}