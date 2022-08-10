#ifndef FLEXFRINGE_INPUTDATALOCATOR_H
#define FLEXFRINGE_INPUTDATALOCATOR_H

#include "input/i_inputdata.h"

class InputDataLocator {
private:
    static IInputData* reader_;

public:
    static void provide(IInputData* reader);
    static IInputData* get();
};


#endif //FLEXFRINGE_INPUTDATALOCATOR_H
