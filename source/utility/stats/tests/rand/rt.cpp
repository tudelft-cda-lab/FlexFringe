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
    print_begin("rt");

    //

    double dof_par = 11.0;

    double t_mean = 0.0;
    double t_var = dof_par / (dof_par - 2.0) ;

    int n_sample = 10000;

    //

    double t_rand = stats::rt(dof_par);
    std::cout << "t rv draw: " << t_rand << std::endl;

    //

#ifdef STATS_TEST_STDVEC_FEATURES
    std::cout << "\n";
    mat_obj t_stdvec = stats::rt<mat_obj>(n_sample,1,dof_par);

    std::cout << "stdvec: t rv mean: " << stats::mat_ops::mean(t_stdvec) << ". Should be close to: " << t_mean << std::endl;
    std::cout << "stdvec: t rv variance: " << stats::mat_ops::var(t_stdvec) << ". Should be close to: " << t_var << std::endl;
#endif

    //

#ifdef STATS_TEST_MATRIX_FEATURES
    std::cout << "\n";
    mat_obj t_vec = stats::rt<mat_obj>(n_sample,1,dof_par);

    std::cout << "Matrix: t rv mean: " << stats::mat_ops::mean(t_vec) << ". Should be close to: " << t_mean << std::endl;
    std::cout << "Matrix: t rv variance: " << stats::mat_ops::var(t_vec) << ". Should be close to: " << t_var << std::endl;
#endif

    //

    std::cout << "\n*** rt: end tests. ***\n" << std::endl;

    return 0;
}
