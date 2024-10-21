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
#include <list>
#include <memory>
#include <unordered_map>
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
    virtual std::unique_ptr<graph_information::graph_base> readline(std::ifstream& input_stream) const;

    virtual std::unique_ptr<apta>
    construct_apta(const std::string_view initial_state,
                   const std::unordered_map<std::string, std::list<std::pair<std::string, std::string>>>& edges,
                   const std::unordered_map<std::string, std::string>& nodes) const = 0;

  public:
    benchmarkparser_base() = default;

    /**
     * @brief Parses the input file, returns a ready apta to be used by the dfa_sul.
     */
    virtual std::unique_ptr<apta> read_input(std::ifstream& input_stream) const;
};

#endif
