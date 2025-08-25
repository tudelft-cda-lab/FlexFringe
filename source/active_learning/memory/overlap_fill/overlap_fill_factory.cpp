/**
 * @file overlap_fill_factory.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2025-05-10
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "overlap_fill_factory.h"

#include "overlap_fill_batch_wise.h"
#include "overlap_fill.h"

#include <stdexcept>

using namespace std;

shared_ptr<overlap_fill_base> overlap_fill_factory::create_overlap_fill_handler(const shared_ptr<sul_base>& sul, string_view ii_name){
  if(ii_name == "overlap_fill_batch_wise")
    return make_shared<overlap_fill_batch_wise>(sul);
  else if(ii_name == "overlap_fill")
    return make_shared<overlap_fill>(sul);
  else
    throw logic_error("Invalid ii_handler name");
}