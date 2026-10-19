#include <math.h>

#include "fastalergia.h"

REGISTER_DEF_DATATYPE(fastalergia_data);
REGISTER_DEF_TYPE(fastalergia);

fastalergia_data::fastalergia_data() : alergia_data (){};

void fastalergia_data::initialize(){
    alergia_data::initialize();
}

/// @brief Implementation of the fast inverse square root algorithm, which uses bitshifts and an iteration of Newton's method
/// @param number the number who's inverse square root should be calculated 
/// @return the inverse square root of number
float fastalergia::fstInvSqrt(float number){
    union{
        float f;
        uint32_t i;
    } conv;

    float x2;
    const float threehalfs = 1.5f;

    x2 = number * 0.5f;
    conv.f = number;
    conv.i = 0x5f3759df - (conv.i >> 1);
    conv.f = conv.f * ( threehalfs - (x2 * conv.f * conv.f));
    return conv.f;

}

/// @brief Instead of the regular alergia bound this uses a sqaured alergia bound eliminating a sqaure root and then uses the fast inverse sqaure root algorithm for the remaining calculations
/// @param right_count 
/// @param left_count 
/// @param right_total 
/// @param left_total 
/// @return difference between the square alergia bound 
double fastalergia::alergia_check(double right_count, double left_count, double right_total, double left_total){

    if(left_count == 0.0 && right_count == 0.0) return 0.0;
    if(left_total == 0.0 || right_total == 0.0) return 0.0;

    double bound = pow((fstInvSqrt(left_total) + fstInvSqrt(right_total)),2);
    bound = bound * 0.5 * log(2.0 / CHECK_PARAMETER);

    double gamma = pow(((left_count + CORRECTION) / left_total) - ((right_count + CORRECTION) / right_total),2);
    if(gamma < 0) gamma = -gamma;

    return bound - gamma;
}