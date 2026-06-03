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

    explicit attribute();
    explicit attribute(const std::string& input);
    explicit attribute(const attribute_info& input);

    void from_string(const std::string &input);

    double get_value(const std::string& val){
        if(discrete){
            if(!r_values.contains(val)) {
                r_values[val] = values.size();
                values.push_back(val);
            }
            return r_values[val];
        }
        double result;
        try {
            result = std::stof(val);
        } catch (const std::invalid_argument&) {
            result = 0.0;
        }
        return result;
    };

    std::string get_name(){
        return name;
    };
    std::string to_string(){
        std::string result = std::string("") +
            (splittable ? "s" : "") +
            (distributionable ? "f" : "") +
            (discrete ? "d" : "") +
            (target ? "t" : "") + "/" +
            name;
        return result;
    };
};

#endif //FLEXFRINGE_ATTRIBUTE_H
