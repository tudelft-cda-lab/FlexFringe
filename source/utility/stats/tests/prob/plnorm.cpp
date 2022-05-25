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

#define TEST_PRINT_PRECISION_1 2
#define TEST_PRINT_PRECISION_2 5

#include "../stats_tests.hpp"

int main()
{
    print_begin("plnorm");

    // parameters

    double mu = 1;
    double sigma = 2;

    //

    std::vector<double> inp_vals = { 2.0,       5.0,        9.0 };
    std::vector<double> exp_vals = { 0.439031,  0.6197099,  0.7252843 };

    //
    // scalar tests

    STATS_TEST_EXPECTED_VAL(plnorm,inp_vals[0],exp_vals[0],false,mu,sigma);
    STATS_TEST_EXPECTED_VAL(plnorm,inp_vals[1],exp_vals[1],true,mu,sigma);
    STATS_TEST_EXPECTED_VAL(plnorm,inp_vals[2],exp_vals[2],false,mu,sigma);

    STATS_TEST_EXPECTED_VAL(plnorm,TEST_NAN,TEST_NAN,false,0,1);                                    // Input NaNs
    STATS_TEST_EXPECTED_VAL(plnorm,0,TEST_NAN,false,TEST_NAN,1);
    STATS_TEST_EXPECTED_VAL(plnorm,0,TEST_NAN,false,0,TEST_NAN);

    STATS_TEST_EXPECTED_VAL(plnorm,0,TEST_NAN,false,0,-1.0);                                        // sigma < 0
    STATS_TEST_EXPECTED_VAL(plnorm,0,TEST_NAN,false,0,TEST_NEGINF);

    STATS_TEST_EXPECTED_VAL(plnorm,1,0.5,false,3,TEST_POSINF);                                      // sigma == +Inf
    STATS_TEST_EXPECTED_VAL(plnorm,1,0.5,false,0,TEST_POSINF);
    STATS_TEST_EXPECTED_VAL(plnorm,3,0.5,false,1,TEST_POSINF);
    STATS_TEST_EXPECTED_VAL(plnorm,TEST_POSINF,1,false,1,TEST_POSINF);
    STATS_TEST_EXPECTED_VAL(plnorm,1,0,false,TEST_POSINF,TEST_POSINF);
    STATS_TEST_EXPECTED_VAL(plnorm,TEST_POSINF,TEST_NAN,false,TEST_POSINF,TEST_POSINF);
    STATS_TEST_EXPECTED_VAL(plnorm,1,0.5,false,1,TEST_POSINF);
    STATS_TEST_EXPECTED_VAL(plnorm,1,1,false,TEST_NEGINF,TEST_POSINF);
    STATS_TEST_EXPECTED_VAL(plnorm,0,0,false,TEST_NEGINF,TEST_POSINF);

    STATS_TEST_EXPECTED_VAL(plnorm,TEST_POSINF,TEST_NAN,false,TEST_POSINF,0);                       // log(x) == mu == Inf
    STATS_TEST_EXPECTED_VAL(plnorm,TEST_POSINF,TEST_NAN,false,TEST_POSINF,1);
    STATS_TEST_EXPECTED_VAL(plnorm,0,0,false,TEST_NEGINF,0);                                        // log(x) == mu == -Inf
    STATS_TEST_EXPECTED_VAL(plnorm,0,0,false,TEST_NEGINF,1);

    STATS_TEST_EXPECTED_VAL(plnorm,TEST_POSINF,1,false,TEST_NEGINF,0);                              // log(x) == Inf, mu == -Inf
    STATS_TEST_EXPECTED_VAL(plnorm,TEST_POSINF,1,false,TEST_NEGINF,1);
    STATS_TEST_EXPECTED_VAL(plnorm,0,0,false,TEST_POSINF,0);                                        // log(x) == -Inf, mu == Inf
    STATS_TEST_EXPECTED_VAL(plnorm,0,0,false,TEST_POSINF,1);

    STATS_TEST_EXPECTED_VAL(plnorm,3,1,false,0,0);                                                  // sigma == 0
    STATS_TEST_EXPECTED_VAL(plnorm,1,0,false,1,0);
    STATS_TEST_EXPECTED_VAL(plnorm,1,1,false,0,0);
    STATS_TEST_EXPECTED_VAL(plnorm,TEST_POSINF,1,false,1,0);
    STATS_TEST_EXPECTED_VAL(plnorm,0,0,false,1,0);
    STATS_TEST_EXPECTED_VAL(plnorm,1,0,false,TEST_POSINF,0);
    STATS_TEST_EXPECTED_VAL(plnorm,1,1,false,TEST_NEGINF,0);

    STATS_TEST_EXPECTED_VAL(plnorm,TEST_POSINF,1,false,0,1);                                        // x == +/-Inf
    STATS_TEST_EXPECTED_VAL(plnorm,TEST_NEGINF,0,false,0,1);

    STATS_TEST_EXPECTED_VAL(plnorm,1,0,false,TEST_POSINF,1);                                        // mu == +/-Inf
    STATS_TEST_EXPECTED_VAL(plnorm,1,1,false,TEST_NEGINF,1);

    //
    // vector/matrix tests

#ifdef STATS_TEST_STDVEC_FEATURES
    STATS_TEST_EXPECTED_MAT(plnorm,inp_vals,exp_vals,std::vector<double>,false,mu,sigma);
    STATS_TEST_EXPECTED_MAT(plnorm,inp_vals,exp_vals,std::vector<double>,true,mu,sigma);
#endif

#ifdef STATS_TEST_MATRIX_FEATURES
    mat_obj inp_mat(2,3);
    inp_mat(0,0) = inp_vals[0];
    inp_mat(1,0) = inp_vals[2];
    inp_mat(0,1) = inp_vals[1];
    inp_mat(1,1) = inp_vals[0];
    inp_mat(0,2) = inp_vals[2];
    inp_mat(1,2) = inp_vals[1];

    mat_obj exp_mat(2,3);
    exp_mat(0,0) = exp_vals[0];
    exp_mat(1,0) = exp_vals[2];
    exp_mat(0,1) = exp_vals[1];
    exp_mat(1,1) = exp_vals[0];
    exp_mat(0,2) = exp_vals[2];
    exp_mat(1,2) = exp_vals[1];

    STATS_TEST_EXPECTED_MAT(plnorm,inp_mat,exp_mat,mat_obj,false,mu,sigma);
    STATS_TEST_EXPECTED_MAT(plnorm,inp_mat,exp_mat,mat_obj,true,mu,sigma);
#endif

    // 

    print_final("plnorm");

    return 0;
}
