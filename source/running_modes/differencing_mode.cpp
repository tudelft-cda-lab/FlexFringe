/**
 * @file differencing_mode.cpp
 * @author Sicco Verwer (s.e.verwer@tudelft.nl)
 * @brief 
 * @version 0.1
 * @date 2024-12-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "parameters.h"
#include "predict_mode.h"
#include "differencing_mode.h"

void differencing_mode::initialize() {
    running_mode_base::initialize();
    read_input_file();

    throw std::runtime_error("not implemented");
    
    if(!APTA_FILE.empty() && !APTA_FILE.empty()){
        std::ifstream input_apta_stream(APTA_FILE);
        std::cerr << "reading apta file - " << APTA_FILE << std::endl;
        the_apta->read_json(input_apta_stream);

        std::ifstream input_apta_stream2(APTA_FILE2);
        std::cerr << "reading apta file - " << APTA_FILE2 << std::endl;
        the_apta2->read_json(input_apta_stream2);

        //std::ostringstream res_stream;
        //std::cout << symmetric_difference(the_apta, the_apta2) << std::endl;
        //delete the_apta2;
    } else {
        throw std::invalid_argument("require two json formatted apta files to compare");
    }
}

int differencing_mode::run() {
    double res = symmetric_difference(the_apta, the_apta2);
    std::cout << res << std::endl;
    return EXIT_SUCCESS;
}

double differencing_mode::difference(apta* apta1, apta* apta2){
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