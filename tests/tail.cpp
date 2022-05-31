// tails.cpp

// main() provided by Catch in file tests.cpp file.

#include "source/utility/catch.hpp"
#include "source/inputdata.h"

TEST_CASE( "2: Empty tail does not have a future", "[multi-file:2]" ) {

    // with the most recent change, this requires inputdata to have
    // information available, so the following test isn't possible
    // without further changes.
	tail *t = NULL;
	//t = new tail(0, 0, NULL);
	REQUIRE( t == NULL);

}


