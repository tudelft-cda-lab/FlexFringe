/**
 * @file positive_traces_sul.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-02-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "parameters.h"
#include "positive_traces_sul.h"
#include "abbadingoparser.h"
//#include "csv.hpp"
#include "csvparser.h"

#include <iostream>

using namespace std;

void positive_traces_sul::preprocessing(){

}

void positive_traces_sul::postprocessing(){

}

void positive_traces_sul::step(){

}

void positive_traces_sul::parse_input(parser& input_parser){

}

positive_traces_sul::positive_traces_sul(){
  ifstream input_stream(INPUT_FILE);  
  cout << "Input file: " << INPUT_FILE << endl;
    
  if(!input_stream) {
      LOG_S(ERROR) << "Input file not found, aborting";
      cerr << "Input file not found, aborting" << endl;
      exit(-1);
  } else {
      cout << "Using input file: " << INPUT_FILE << endl;
  }

  bool read_csv = false;
  if(INPUT_FILE.compare(INPUT_FILE.length() - 4, INPUT_FILE.length(), ".csv") == 0){
      read_csv = true;
  }

  if(read_csv) {
      auto input_parser = csv_parser(input_stream, csv::CSVFormat().trim({' '}));
      
  } else {
      auto input_parser = abbadingoparser(input_stream);
      
  }
}