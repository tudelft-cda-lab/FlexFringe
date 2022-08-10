#include <cassert>
#include "inputdatalocator.h"

IInputData* InputDataLocator::reader_;

void InputDataLocator::provide(IInputData *reader) {
    InputDataLocator::reader_ = reader;
}

IInputData* InputDataLocator::get() {
    assert(InputDataLocator::reader_ != nullptr);
    return InputDataLocator::reader_;
}
