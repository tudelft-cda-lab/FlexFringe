//
// Created by sicco on 23/04/2021.
//

#include "parameters.h"
#include "predict.h"
#include "differencing.h"

double difference(apta* apta1, apta* apta2){
    double kl_diff = 0.0;
    for(int i = 0; i < DIFF_SIZE; ++i){
        double score1 = 0.0;
        double score2 = 0.0;
        int length = 1;
        apta_node* n1 = apta1->get_root();
        apta_node* n2 = apta2->get_root();
        tail* t = n1->get_data()->sample_tail();
        //cerr << t->to_string() << endl;
        while(!t->is_final() && length < DIFF_MAX_LENGTH){
            double kl_score1 = std::max(compute_score(n1, t), DIFF_MIN);
            double kl_score2 = std::max(compute_score(n2, t), DIFF_MIN);
            score1 += exp(kl_score1) * kl_score1;
            score2 += exp(kl_score1) * kl_score2;
            n1 = single_step(n1, t, apta1);
            n2 = single_step(n2, t, apta2);
            if(n1 == nullptr || n2 == nullptr) break;
            mem_store::delete_tail(t);
            t = n1->get_data()->sample_tail();
            //cerr << t->to_string() << " " << n1 << " " << n2 << endl;
            length++;
        }
        if(FINAL_PROBABILITIES && t->is_final()){
            score1 += std::max(compute_score(n1, t), DIFF_MIN);
            score2 += std::max(compute_score(n2, t), DIFF_MIN);
        }
        mem_store::delete_tail(t);

        //cerr << score1 << " " << score2 << endl;

        kl_diff += (score1 - score2) / (double)length;

        //cerr << kl_diff << endl;
    }
    return kl_diff;
}

double symmetric_difference(apta* a1, apta* a2){
    return difference(a1, a2) + difference(a2, a1);
}