#ifndef _FACTORY_H_
#define _FACTORY_H_

#define REGISTER_DEF_TYPE(NAME) \
    DerivedRegister<NAME> NAME::reg(#NAME)

#define REGISTER_DEF_DATATYPE(NAME) \
    DerivedDataRegister<NAME> NAME::reg(#NAME)

#include "evaluate.h"

// shorthands for data and evaluation function objects
template<typename T> evaluation_data * createDataT() { return new T; }
template<typename T> evaluation_function * createT() { return new T; }

// types for storing the pointers to the objects to be created    
typedef std::map<std::string, evaluation_data*(*)()> map_datatype;
typedef std::map<std::string, evaluation_function*(*)()> map_type;

// public classes each for
// evaluation data objects
struct BaseDataFactory {

    static evaluation_data * createInstance(std::string const& s) {
        map_datatype::iterator it = getMap()->find(s);
        if(it == getMap()->end())
            return 0;
        return it->second();
    }

public:
    static map_datatype * getMap() {
        // never delete'ed. (exist until program termination)
        // because we can't guarantee correct destruction order 
        if(!map) { map = new map_datatype; } 
        return map; 
    }

private:
    static map_datatype* map;

};

// as well as evaluation function objects
struct BaseFactory {

    static evaluation_function * createInstance(std::string const& s) {
        map_type::iterator it = getMap()->find(s);
        if(it == getMap()->end())
            return 0;
        return it->second();
    }

public:
    static map_type * getMap() {
        // never delete'ed. (exist until program termination)
        // because we can't guarantee correct destruction order 
        if(!map) { map = new map_type; } 
        return map; 
    }

private:
    static map_type* map;


};

// create derived instances of it for each
// evaluation data objects
template<typename T>
struct DerivedDataRegister : BaseDataFactory { 
    DerivedDataRegister(std::string const& s) { 
        getMap()->insert(std::make_pair(s, &createDataT<T>));
    }
};


// and evaluation function objects
template<typename T>
struct DerivedRegister : BaseFactory { 
    DerivedRegister(std::string const& s) { 
        getMap()->insert(std::make_pair(s, &createT<T>));
    }
};




#endif
