#ifndef __FASTALERGIA__
#define __FASTALERGIA__

#include "alergia.h"

/* The data contained in every node of the prefix tree or DFA */
class fastalergia_data: public alergia_data {
    protected:
        REGISTER_DEC_DATATYPE(fastalergia_data);
    public:

        fastalergia_data();
        virtual void initialize();
};

class fastalergia : public alergia{

    protected:
        REGISTER_DEC_TYPE(fastalergia);
    public:

        static float fstInvSqrt(float number);

        virtual double alergia_check(double right_count, double left_count, double right_total, double left_total);

};

#endif