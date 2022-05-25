
#ifndef _ENSEMBLE_H_
#define _ENSEMBLE_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <list>
#include "state_merger.h"
#include "refinement.h"

using namespace std;

void bagging(state_merger* merger, string output_file, int nr_estimators);

#endif /* _ENSEMBLE_H_ */
