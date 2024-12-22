/**
 * @file regex_mode.cpp
 * @author Hielke Walinga
 * @brief 
 * @version 0.1
 * @date 2024-12-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "regex_mode.h"
#include "regex_builder.h"
#include "inputdatalocator.h" 

#include <iostream>
#include <ranges>

void regex_mode::initialize(){
  running_mode_base::initialize();
}

int regex_mode::run(){
  std::ifstream input_apta_stream(APTA_FILE);
  LOG_S(INFO) << "Reading apta file - " << APTA_FILE;
  the_apta->read_json(input_apta_stream);
  LOG_S(INFO) << "Finished reading apta file.";
  auto coloring = std::make_tuple(PRINT_RED, PRINT_BLUE, PRINT_WHITE);
  regex_builder builder = regex_builder(*the_apta, *merger, coloring, psql::db::num2str);
  LOG_S(INFO) << "Finished building the regex builder";
  auto delimiter = "\t";
  for (std::string type : std::views::keys(inputdata_locator::get()->get_r_types())) {
    for (auto regex : builder.to_regex(type, 1)){
      /* cout << "Got regex with size: " << regex.size() << endl; */
      std::cout << type << delimiter << regex << std::endl;
    }
  }
  return EXIT_SUCCESS;
}