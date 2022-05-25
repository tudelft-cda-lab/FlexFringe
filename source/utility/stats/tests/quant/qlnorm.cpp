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

#define STATS_TEST_INPUT_TYPE 1

#define TEST_PRINT_PRECISION_1 2
#define TEST_PRINT_PRECISION_2 5

#include "../stats_tests.hpp"

int main()
{
    print_begin("qlnorm");

    // parameters

    double mu = 1;
    double sigma = 2;

    double exp_mu = std::exp(mu);

    //

    std::vector<double> inp_vals = { 0.15,       0.5,       0.79 };
    std::vector<double> exp_vals = { 0.3420266,  2.718282,  13.63776 };

    //
    // scalar tests

    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,inp_vals[0],exp_vals[0],mu,sigma);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,inp_vals[1],exp_vals[1],mu,sigma);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,inp_vals[2],exp_vals[2],mu,sigma);

    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0.99,285.0589,1,2);

    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,TEST_NAN,TEST_NAN,0,1);                                    // Input NaNs
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0,TEST_NAN,TEST_NAN,1);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0,TEST_NAN,0,TEST_NAN);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,TEST_NAN,TEST_NAN,TEST_NAN,TEST_NAN);

    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,-0.1,TEST_NAN,mu,sigma);                                   // p < 0 or p > 1 cases
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm, 1.1,TEST_NAN,mu,sigma);

    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0.0,0.0,mu,sigma);                                         // p == 0
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0.0,exp_mu,mu,0);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0.0,TEST_NAN,TEST_POSINF,sigma);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0.0,0.0,TEST_NEGINF,sigma);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0.0,TEST_NAN,mu,TEST_POSINF);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0.0,TEST_NAN,TEST_POSINF,TEST_POSINF);

    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,1.0,TEST_POSINF,mu,sigma);                                 // p == 1
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,1.0,exp_mu,mu,0);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,1.0,TEST_POSINF,TEST_POSINF,sigma);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,1.0,TEST_NAN,TEST_NEGINF,sigma);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,1.0,TEST_NAN,mu,TEST_POSINF);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,1.0,TEST_NAN,TEST_POSINF,TEST_POSINF);

    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0,exp_mu,mu,0.0);                                          // sigma <= 0
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0,TEST_NAN,0,-1.0);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0,TEST_NAN,0,TEST_NEGINF);

    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0,TEST_NAN,1,TEST_POSINF);                                 // sigma == +Inf
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,TEST_POSINF,TEST_NAN,1,TEST_POSINF);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0,TEST_NAN,TEST_POSINF,TEST_POSINF);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,TEST_POSINF,TEST_NAN,TEST_POSINF,TEST_POSINF);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,TEST_NEGINF,TEST_NAN,1,TEST_POSINF);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0,TEST_NAN,TEST_NEGINF,TEST_POSINF);
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,TEST_NEGINF,TEST_NAN,TEST_NEGINF,TEST_POSINF);

    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0.5,TEST_POSINF,TEST_POSINF,1);                            // mu == Inf
    STATS_TEST_EXPECTED_QUANT_VAL(qlnorm,0.5,0.0,TEST_NEGINF,1);                                    // mu == -Inf

    //
    // vector/matrix tests

#ifdef STATS_TEST_STDVEC_FEATURES
    STATS_TEST_EXPECTED_QUANT_MAT(qlnorm,inp_vals,exp_vals,std::vector<double>,mu,sigma);
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

    STATS_TEST_EXPECTED_QUANT_MAT(qlnorm,inp_mat,exp_mat,mat_obj,mu,sigma);
#endif

    // 

    print_final("qlnorm");

    return 0;
}
