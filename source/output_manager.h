/**
 * @file output_manager.cpp
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief Two classes to manage the input and output of the framework.
 * @version 0.1
 * @date 2024-12-31
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef _OUTPUT_MANAGER_H_
#define _OUTPUT_MANAGER_H_

#include "state_merger.h"
#include "parameters.h"

#include <string>
#include <string_view>

/**
 * @brief Manages the output path.
 * 
 */
class output_manager {
  private:
    inline static std::string outfile_path = ""; 

  public:
    output_manager() = delete;

    static void init_outfile_path();
    static std::string_view get_outfile_path();

    static void print_final_automaton(state_merger* merger, const std::string& append_string);
    static void print_current_automaton(state_merger* merger, const std::string& output_file, const std::string& append_string);

};

#endif // _IO_MANAGER_H_