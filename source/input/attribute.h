#ifndef FLEXFRINGE_ATTRIBUTE_H
#define FLEXFRINGE_ATTRIBUTE_H


#include <vector>
#include <string>
#include <map>
#include <stdexcept>
#include "input/parsers/attribute_info.h"

/**
 * @brief Wrapper class for the input data. Supports functionalities
 * such as alphabet functions, file transformations and data added to the APTA.
 *
 */
class attribute{
public:
    bool discrete;
    bool splittable;
    bool distributionable;
    bool target;

    std::vector<std::string> values;
    std::map<std::string, int> r_values;

    std::string name;

    explicit attribute(const std::string& input);
    explicit attribute(const attribute_info& input);

    inline double get_value(std::string val){
        if(discrete){
            if(r_values.find(val) == r_values.end()) {
                r_values[val] = values.size();
                values.push_back(val);
            }
            return (double) r_values[val];
        } else {
            double result;
            try {
                result = std::stof(val);
            } catch (const std::invalid_argument& e) {
                result = 0.0;
            }
            return result;
        }
    };

    inline std::string get_name(){
        return name;
    };

};

#endif //FLEXFRINGE_ATTRIBUTE_H
