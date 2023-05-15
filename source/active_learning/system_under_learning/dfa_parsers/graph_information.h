/**
 * @file graph_information.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-04-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef _GRAPH_INFORMATION_H_
#define _GRAPH_INFORMATION_H_

#include <string>

namespace graph_information {
  
  /**
   * @brief Used for polymorphism.
   * 
   */
  struct graph_base {
    public:
     graph_base() = default; 

     virtual ~graph_base() = default;
  };

  /**
   * @brief Does not represent graph, but indicates that the read line is a header.
   * 
   */
  struct header_line : public graph_base {
    public:
      header_line() = default;
  };

  /**
   * @brief Represents transition.
   * 
   */
  struct transition_element : public graph_base {
    public:

      std::string s1, s2; // two states
      std::string symbol;
      std::string data;

      transition_element() = default;
  };

  /**
   * @brief For initial states. Carries information about which state to hit in flexfringe. Not supported yet.
   * An initial transition looks e.g. like this: __start0 -> s1, where __start0 is the identifier of the initial 
   * information. E.g. when multiple starting states are possible which one to start with.
   * 
   */
  struct initial_transition : public graph_base {
    public: 
      std::string start_id; // a key to find indentifier
      std::string state;

      initial_transition() = default;
  };

  /**
   * @brief Carries the information of initial transitions.
   * 
   */
  struct initial_transition_information : public graph_base {
    public:
      std::string start_id; // matches the start_id in initial_transition struct
      std::string symbol;
      std::string data;

      initial_transition_information() = default;
  };

  /**
   * @brief Obviously a state.
   * 
   */
  struct graph_node : public graph_base {
    public:
      std::string id;
      std::string shape;

      graph_node() = default;
  };

}
#endif