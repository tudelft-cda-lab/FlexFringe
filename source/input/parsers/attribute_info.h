#ifndef FLEXFRINGE_ATTRIBUTE_INFO_H
#define FLEXFRINGE_ATTRIBUTE_INFO_H


#include <array>
#include <string>
#include <set>
#include <utility>

class attribute_info {
private:
    enum attribute_types {
        discrete,
        splittable,
        distributionable,
        target,

        // Last entry used to know total number of types at compile time
        count
    };

    std::array<bool, count> types {};
    std::string name;
    std::string value;

    void set_discrete(bool val) { types[discrete] = val; }
    void set_splittable(bool val) { types[splittable] = val; }
    void set_distributionable(bool val) { types[distributionable] = val; }
    void set_target(bool val) { types[target] = val; }

public:
    attribute_info(std::string name,
                   std::string value,
                   const std::set<char>& type_set)
                   : name(std::move(name))
                   , value(std::move(value))
    {
        set_discrete(type_set.contains('d'));
        set_splittable(type_set.contains('s'));
        set_distributionable(type_set.contains('f'));
        set_target(type_set.contains('t'));
    }

    // Copy constructor and clone to use prototype pattern
    attribute_info(const attribute_info& other) = default;
    attribute_info clone(std::string new_value) {
        attribute_info new_info = attribute_info(*this);
        new_info.value = std::move(new_value);
        return new_info;
    }

    [[nodiscard]] bool is_discrete() const { return types[discrete]; }
    [[nodiscard]] bool is_splittable() const { return types[splittable]; }
    [[nodiscard]] bool is_distributionable() const { return types[distributionable]; }
    [[nodiscard]] bool is_target() const { return types[target]; }

    [[nodiscard]] const std::string& get_name() const { return name; }
    [[nodiscard]] const std::string& get_value() const { return value; }

    void set_value(std::string val) {value = std::move(val);}

};


#endif //FLEXFRINGE_ATTRIBUTE_INFO_H
