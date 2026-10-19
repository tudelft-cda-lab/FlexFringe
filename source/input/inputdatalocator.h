#ifndef FLEXFRINGE_INPUTDATALOCATOR_H
#define FLEXFRINGE_INPUTDATALOCATOR_H

#include "input/inputdata.h"

class inputdata_locator {
private:
    inline static inputdata* reader_ = nullptr;

public:
    static void provide(inputdata* reader);
    static inputdata* get();
};


#endif //FLEXFRINGE_INPUTDATALOCATOR_H
