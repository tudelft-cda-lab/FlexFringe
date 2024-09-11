#include "state_merger.h"
#include "evaluate.h"
#include "evaluation_factory.h"
#include "parameters.h"
#include "paul_heuristic.h"
#include "input/inputdatalocator.h"

#include <map>
#include <set>
#include <unordered_set>

REGISTER_DEF_TYPE(paul_heuristic);
REGISTER_DEF_DATATYPE(paul_data);