/**
 * @file benchmark_dfaparser.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief A class to connect to the .dot files given by https://automata.cs.ru.nl/Syntax/Overview
 * IMPORTANT: Only DFAs supported in this moment.
 * @version 0.1
 * @date 2023-04-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _BENCHMARK_DFAPARSER_H_
#define _BENCHMARK_DFAPARSER_H_

#include "benchmarkparser_base.h"

/**
 * @brief This base class parses the files from the benchmark given by https://automata.cs.ru.nl/Syntax/Overview
 * For now we only implement the DFAs.
 * 
 */
class benchmark_dfaparser : public benchmarkparser_base {
protected:
    virtual std::unique_ptr<apta> read_input(ifstream& input_stream) const override;
    //virtual std::unique_ptr<graph_base> readline(ifstream& input_stream) const override;
    
public:
    benchmark_dfaparser() = default;
};


#endif
