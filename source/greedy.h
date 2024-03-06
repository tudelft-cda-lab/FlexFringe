
#ifndef _RANDOM_GREEDY_H_
#define _RANDOM_GREEDY_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <list>
#include "state_merger.h"
#include "refinement.h"

const int RANDOMG = 1;
const int NORMALG = 2;

void greedy_run(state_merger* merger);

#endif /* _RANDOM_GREEDY_H_ */
