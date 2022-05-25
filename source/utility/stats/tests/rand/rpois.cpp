/*################################################################################
  ##
  ##   Copyright (C) 2011-2020 Keith O'Hara
  ##
  ##   This file is part of the StatsLib C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

#include "../stats_tests.hpp"

int main()
{
    print_begin("rpois");

    //

    double rate_par = 5.0;

    double pois_mean = rate_par;
    double pois_var = rate_par;

    int n_sample = 10000;

    //

    double pois_rand = stats::rpois(rate_par);
    std::cout << "pois rv draw: " << pois_rand << std::endl;

    //

#ifdef STATS_TEST_STDVEC_FEATURES
    std::cout << "\n";
    std::vector<double> pois_stdvec = stats::rpois<std::vector<double>>(n_sample,1,rate_par);

    std::cout << "stdvec: pois rv mean: " << stats::mat_ops::mean(pois_stdvec) << ". Should be close to: " << pois_mean << std::endl;
    std::cout << "stdvec: pois rv variance: " << stats::mat_ops::var(pois_stdvec) << ". Should be close to: " << pois_var << std::endl;
#endif

    //

#ifdef STATS_TEST_MATRIX_FEATURES
    std::cout << "\n";
    mat_obj pois_vec = stats::rpois<mat_obj>(n_sample,1,rate_par);

    std::cout << "Matrix: pois rv mean: " << stats::mat_ops::mean(pois_vec) << ". Should be close to: " << pois_mean << std::endl;
    std::cout << "Matrix: pois rv variance: " << stats::mat_ops::var(pois_vec) << ". Should be close to: " << pois_var << std::endl;
#endif

    //

    std::cout << "\n*** rpois: end tests. ***\n" << std::endl;

    return 0;
}
