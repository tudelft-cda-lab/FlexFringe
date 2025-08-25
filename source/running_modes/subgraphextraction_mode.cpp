/**
 * @file subgraphextraction_mode.cpp
 * @author Sicco Verwer (s.e.verwer@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "subgraphextraction_mode.h"
#include "csv.hpp"
#include "common.h"

#include "input/abbadingoreader.h"
#include "input/parsers/abbadingoparser.h"
#include "input/parsers/csvparser.h"

#include <fstream>

void subgraphextraction_mode::initialize(){
  // TODO: do this one
}

int subgraphextraction_mode::run(){
  std::ifstream input_stream(INPUT_FILE);    
  if(!input_stream) {
      LOG_S(ERROR) << "Input file not found, aborting";
      std::cerr << "Input file not found, aborting" << std::endl;
      exit(-1);
  } else {
      std::cout << "Using input file: " << INPUT_FILE << std::endl;
  }

  if(!APTA_FILE.empty()){
    std::ifstream input_apta_stream(APTA_FILE);
    std::cout << "reading apta file - " << APTA_FILE << std::endl;
    the_apta->read_json(input_apta_stream);
    std::cout << "Finished reading apta file." << std::endl;

    bool read_csv = false;
    if(INPUT_FILE.ends_with(".csv")){
        read_csv = true;
    }

    if(read_csv) {
        auto input_parser = csv_parser(input_stream, csv::CSVFormat().trim({' '}));
        id.read(&input_parser);
    } else {
        auto input_parser = abbadingoparser(input_stream);
        //id.read(&input_parser);
        std::cout << "starting to predict" << std::endl;
        auto strategy = in_order();

        state_set visited_nodes;
        std::unordered_set<apta_guard*> traversed_guards;
        for (auto* tr : id.trace_iterator(input_parser, strategy)) {    
            apta_node* n = merger->get_aut()->get_root();
            tail* t = tr->get_head();

            for(int j = 0; j < t->get_length(); j++){
               apta_guard* g = n->guard(t->get_symbol());
               traversed_guards.insert(g);

                n = single_step(n, t, merger->get_aut());
                if(n == nullptr) break;
                t = t->future();
                visited_nodes.insert(n);
            }
            tr->erase();
        }

        std::ofstream output(OUTPUT_FILE + ".part.dot");

        std::stringstream dot_output_buf;
        merger->get_aut()->print_dot(dot_output_buf, &visited_nodes, &traversed_guards);
        std::string dot_output = "// produced with flexfringe // " + COMMAND + '\n'+ dot_output_buf.str();

      output << dot_output;
      output.close();
    }
    } else {
        throw std::invalid_argument("require a json formatted apta file to make predictions");
    }
}