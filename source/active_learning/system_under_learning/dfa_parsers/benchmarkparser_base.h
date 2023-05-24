/**
 * @file benchmarkparser_base.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Base class to parser from the benchmark given by https://automata.cs.ru.nl/Syntax/Overview
 * 
 * @version 0.1
 * @date 2023-04-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _BENCHMARKPARSER_BASE_H_
#define _BENCHMARKPARSER_BASE_H_

#include "apta.h"
#include "graph_information.h"

#include <fstream>
#include <memory>
#include <list>
#include <utility>

/**
 * @brief This base class parses the files from the benchmark given by https://automata.cs.ru.nl/Syntax/Overview
 * For now we only implement the DFAs.
 * 
 */
class benchmarkparser_base {
protected:
    /** 
    * @return std::unique_ptr<apta> The ready apta.
    */
    virtual std::unique_ptr<graph_information::graph_base> readline(ifstream& input_stream) const;
    
public:
    benchmarkparser_base() = default;

    /**
    * @brief Parses the input file, returns a ready apta to be used by the dfa_sul.
    */
    virtual std::unique_ptr<apta> read_input(ifstream& input_stream) const = 0;
};


#endif
