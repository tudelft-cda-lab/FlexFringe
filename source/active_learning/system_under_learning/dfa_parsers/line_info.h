/**
 * @file line_info.h
 * @author Robert Baumgartner (r.baumgartner-1@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2023-04-20
 * 
 * @copyright Copyright (c) 2023
 * 
 */

/**
 * @brief Enum class to identify the representation of a line. Helps raise legibility.
 * 
 */
enum class linetype {
  initial_state,
  state,
  transition,
} 

/**
 * @brief A struct that contains info about a read line. Will be used to construct the apta.
 * 
 */
struct line_info {
   public:
    bool is_header;

    line_info() {
      is_header = false;

    }
};