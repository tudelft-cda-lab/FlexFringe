
#ifndef _ENSEMBLE_H_
#define _ENSEMBLE_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <list>
#include "state_merger.h"
#include "refinement.h"


void bagging(state_merger* merger, std::string output_file, int nr_estimators);

#endif /* _ENSEMBLE_H_ */
