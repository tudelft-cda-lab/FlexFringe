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
    print_begin("dexp");

    // parameters

    double rate = 2.0;

    //

    std::vector<double> inp_vals = { 1.0,        2.0,         3.0 };
    std::vector<double> exp_vals = { 0.2706706,  0.03663128,  0.004957504 };

    //
    // scalar tests

    STATS_TEST_EXPECTED_VAL(dexp,inp_vals[0],exp_vals[0],false,rate);
    STATS_TEST_EXPECTED_VAL(dexp,inp_vals[1],exp_vals[1],true,rate);
    STATS_TEST_EXPECTED_VAL(dexp,inp_vals[2],exp_vals[2],false,rate);

    STATS_TEST_EXPECTED_VAL(dexp,TEST_NAN,TEST_NAN,false,3);                                        // Input NaNs
    STATS_TEST_EXPECTED_VAL(dexp,1,TEST_NAN,false,TEST_NAN);

    STATS_TEST_EXPECTED_VAL(dexp,1,0,false,0);                                                      // rate <= 0
    STATS_TEST_EXPECTED_VAL(dexp,0,0,false,0);
    STATS_TEST_EXPECTED_VAL(dexp,1,TEST_NAN,false,-1.0);
    STATS_TEST_EXPECTED_VAL(dexp,1,TEST_NAN,false,TEST_NEGINF);

    STATS_TEST_EXPECTED_VAL(dexp,1,TEST_NAN,false,TEST_POSINF);                                     // rate == Inf

    STATS_TEST_EXPECTED_VAL(dexp,0,0,false,0);                                                      // x == 0
    STATS_TEST_EXPECTED_VAL(dexp,0,1,false,1);

    STATS_TEST_EXPECTED_VAL(dexp,TEST_POSINF,TEST_NAN,false,0);                                     // x == Inf
    STATS_TEST_EXPECTED_VAL(dexp,TEST_POSINF,0,false,1);
    STATS_TEST_EXPECTED_VAL(dexp,TEST_POSINF,TEST_NAN,false,TEST_POSINF);
    STATS_TEST_EXPECTED_VAL(dexp,TEST_NEGINF,0,false,2);                                            // x == -Inf
    STATS_TEST_EXPECTED_VAL(dexp,TEST_NEGINF,TEST_NAN,false,TEST_NEGINF);

    //
    // vector/matrix tests

#ifdef STATS_TEST_STDVEC_FEATURES
    STATS_TEST_EXPECTED_MAT(dexp,inp_vals,exp_vals,std::vector<double>,false,rate);
    STATS_TEST_EXPECTED_MAT(dexp,inp_vals,exp_vals,std::vector<double>,true,rate);
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

    STATS_TEST_EXPECTED_MAT(dexp,inp_mat,exp_mat,mat_obj,false,rate);
    STATS_TEST_EXPECTED_MAT(dexp,inp_mat,exp_mat,mat_obj,true,rate);
#endif

    // 

    print_final("dexp");

    return 0;
}
