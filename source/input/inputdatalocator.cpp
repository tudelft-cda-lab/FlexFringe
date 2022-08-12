#include <cassert>
#include "inputdatalocator.h"

inputdata* inputdata_locator::reader_;

void inputdata_locator::provide(inputdata *reader) {
    inputdata_locator::reader_ = reader;
}

inputdata* inputdata_locator::get() {
    assert(inputdata_locator::reader_ != nullptr);
    return inputdata_locator::reader_;
}
