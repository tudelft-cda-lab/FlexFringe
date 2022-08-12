//
// Created by sicco on 23/04/2021.
//

#include "predict.h"
#include <queue>

struct tail_state_compare{ bool operator()(const pair<double, pair<apta_node*, tail*>> &a, const pair<double, pair<apta_node*, tail*>> &b) const{ return a.first < b.first; } };

int rownr = 1;
map<int,double> sw_score_per_symbol;
map<tail*,double> sw_individual_tail_score;

double compute_jump_penalty(apta_node* old_node, apta_node* new_node){
    if(ALIGN_DISTANCE_PENALTY != 0) return ALIGN_DISTANCE_PENALTY * (double)(old_node->merged_apta_distance(new_node, -1));
    return 0.0;
}

double compute_score(apta_node* next_node, tail* next_tail){
    if(PREDICT_ALIGN){ return next_node->get_data()->align_score(next_tail); }
    return next_node->get_data()->predict_score(next_tail);
}

double update_score(double old_score, apta_node* next_node, tail* next_tail){
    double score = compute_score(next_node, next_tail);
    if(PREDICT_MINIMUM) return min(old_score, score);
    return old_score + score;
}

list<int> state_sequence;
list<double> score_sequence;
list<bool> align_sequence;
apta_node* ending_state = nullptr;
tail* ending_tail = nullptr;

void align(state_merger* m, tail* t, bool always_follow, double lower_bound) {
    apta_node *n = m->get_aut()->get_root();

    double score;

    priority_queue<pair<double, pair<apta_node *, tail *>>,
            vector<pair<double, pair<apta_node *, tail *>>>, tail_state_compare> Q;
    map<apta_node *, map<int, double> > V;

    state_set *states = m->get_all_states();

    Q.push(pair<double, pair<apta_node *, tail *>>(log(1.0),
                                                        pair<apta_node *, tail *>(n, t)));

    apta_node *next_node = nullptr;
    tail *next_tail = nullptr;

    while (!Q.empty()) {
        pair<double, pair<apta_node *, tail *>> next = Q.top();
        Q.pop();

        score = next.first;
        next_node = next.second.first;
        next_tail = next.second.second;

        //cerr << score << endl;

        if (next_tail == nullptr) {
            break;
        }
        if (lower_bound != 1 && score < lower_bound){
            break;
        }

        int index = next_tail->get_index();
        if (V.find(next_node) != V.end() && V[next_node].rbegin()->first >= index) continue;
        V[next_node][index] = score;

        //cerr << index << endl;

        if (next_tail->is_final()) {
            // STOP RUN
            //cerr << "final: " << compute_score(score, next_node, next_tail) << endl;
            Q.push(pair<double, pair<apta_node *, tail *>>(
                    update_score(score, next_node, next_tail),
                    pair<apta_node *, tail *>(pair<apta_node *, tail *>(next_node, 0))));
        } else {
            // FOLLOW TRANSITION
            apta_node *child = next_node->child(next_tail);
            if (child != nullptr) {
                child = child->find();
                //cerr << "follow: " << compute_score(score, next_node, next_tail) << endl;
                Q.push(pair<double, pair<apta_node *, tail *>>(
                        update_score(score, next_node, next_tail),
                        pair<apta_node *, tail *>(child, next_tail->future())));
            }

            if (always_follow && child != nullptr) continue;

            // JUMP TO ALIGN -- calling align consistent
            for(auto jump_child : *states){
                if (jump_child->get_data()->align_consistent(next_tail)) {
                    //apta_node *next_child = jump_child->child(next_tail)->find();
                    //cerr << "jump: " << compute_score(score, next_node, next_tail) << endl;
                    Q.push(pair<double, pair<apta_node *, tail *>>(
                            update_score(score, next_node, next_tail) +
                                    compute_jump_penalty(next_node, jump_child),
                            pair<apta_node *, tail *>(jump_child, next_tail)));
                }
            }

            // SKIP TO ALIGN
            // UNCLEAR whether this is needed.
            //cerr << "skip: " << compute_score(score, next_node, next_tail) << endl;
            /* Q.push(pair<double, pair<apta_node *, tail *>>(
                    compute_score(score, next_node, next_tail),
                    pair<apta_node *, tail *>(next_node, next_tail->future())));
            */
        }
    }

    // RUN NOT ENDED
    if(next_tail != nullptr || score == -INFINITY){
        return;
    }

    ending_state = next_node;

    double current_score = score;
    tail* current_tail = t;
    while(current_tail->future() != nullptr){ current_tail = current_tail->future(); }

    ending_tail = current_tail;
    if(ending_tail->is_final()) ending_tail = ending_tail->past();

    //state_sequence->push_front(next_node->number);
    //score_sequence->push_front(next_node->data->align_score(current_tail));
    //current_score -= log(next_node->data->predict_score(current_tail));
    //current_tail = current_tail->past();

    while(current_tail != nullptr){
        //cerr << "current score : " << current_score << endl;
        int index = current_tail->get_index();

        //cerr << index << endl;

        apta_node* prev_node = nullptr;
        double max_score = -1;
        bool advance = true;
        for(auto node : *states){
            if (V.find(node) != V.end()) {
                map<int, double> vm = V[node];
                if (vm.find(index) != vm.end()) {
                    double old_score = vm[index];
                    if (current_tail->is_final()){
                        double score = update_score(old_score, node, current_tail);
                        //cerr << "final " << old_score << " " << score << endl;
                        if (score == current_score) {
                            max_score = old_score;
                            prev_node = node;
                            advance = true;
                            break;
                        }
                    }
                    if (node->child(current_tail) != 0 && node->child(current_tail)->find() == next_node) {
                        double score = update_score(old_score, node, current_tail);
                        //cerr << "take transition " << old_score << " " << score << endl;
                        if (score == current_score) {
                            max_score = old_score;
                            prev_node = node;
                            advance = true;
                            break;
                        }
                    } /* else if (node == next_node) {
                        double score = compute_score(old_score, node, current_tail);
                        //cerr << "skip symbol " << old_score << " " << score << endl;
                        if (score == current_score) {
                            max_score = old_score;
                            prev_node = node;
                            advance = true;
                            break;
                        }
                    } */
                }
                if (vm.find(index+1) != vm.end() && !current_tail->is_final()){
                    double old_score = vm[index+1];
                    double score = update_score(old_score, node, current_tail->future());
                    //cerr << "jump " << old_score << " " << score << endl;
                    if (score == current_score) {
                        max_score = old_score;
                        prev_node = node;
                        advance = false;
                    }
                }
            }
        }

        if(prev_node != nullptr) {
            state_sequence.push_front(next_node->get_number());
            score_sequence.push_front(prev_node->get_data()->align_score(current_tail));
            align_sequence.push_front(advance);
            current_score = max_score;
            next_node = prev_node;
            if(advance) current_tail = current_tail->past();
            //cerr << current_score << endl;
        }
    }
}

double prob_single_parallel(tail* p, tail* t, apta_node* n, double prod_prob, bool flag){
    double result = -100000;
    if(n == nullptr) return result;
    if(t == p) t = t->future();
    n = n->find();

    if(t->is_final()){
        if(flag){
            if(FINAL_PROBABILITIES){
                return prod_prob + log(n->get_data()->predict_score(t));
            }
            return prod_prob;
        }
        double prob = n->get_data()->predict_score(p);
        return prob_single_parallel(p, t, n->child(p), prod_prob + log(prob), true);
    } else {
        double prob = n->get_data()->predict_score(t);
        result = prob_single_parallel(p, t->future(), n->child(t), prod_prob + log(prob), flag);
        if(result != -100000) return result;

        if (!flag) {
            double prob = n->get_data()->predict_score(p);
            return prob_single_parallel(p, t, n->child(p), prod_prob + log(prob), true);
        }
    }
    return -100000;
}

apta_node* single_step(apta_node* n, tail* t, apta* a){
    apta_node* child = n->child(t);
    if(child == 0){
        if(PREDICT_RESET) return a->get_root();
        else if(PREDICT_REMAIN) return n;
        else return nullptr;
    }
    return child->find();
}

double predict_trace(state_merger* m, trace* tr){
    apta_node* n = m->get_aut()->get_root();
    tail* t = tr->get_head();
    double score = 0.0;

    for(int j = 0; j < t->get_length(); j++){
        score = compute_score(n, t);
        n = single_step(n, t, m->get_aut());
        if(n == nullptr) break;

        t = t->future();
    }

    if(FINAL_PROBABILITIES && t->get_symbol() == -1){
        score = compute_score(n, t);
    }
    return score;
}

double add_visits(state_merger* m, trace* tr){
    apta_node* n = m->get_aut()->get_root();
    tail* t = tr->get_head();
    double score = 0.0;

    for(int j = 0; j < t->get_length(); j++){
        n = single_step(n, t, m->get_aut());
        if(n == nullptr) break;
        t = t->future();
    }
    return score;
}

void predict_trace_update_sequences(state_merger* m, tail* t){
    apta_node* n = m->get_aut()->get_root();
    double score = 0.0;

    for(int j = 0; j < t->get_length(); j++){
        score = compute_score(n, t);
        score_sequence.push_back(score);

        n = single_step(n, t, m->get_aut());
        if(n == nullptr) break;

        t = t->future();
        state_sequence.push_back(n->get_number());
        align_sequence.push_back(true);
    }

    if(FINAL_PROBABILITIES && t->get_symbol() == -1){
        score = compute_score(n, t);
        score_sequence.push_back(score);
        state_sequence.push_back(n->get_number());
        align_sequence.push_back(true);
    }

    ending_state = n;
    ending_tail = t;
    if(ending_tail->is_final()) ending_tail = ending_tail->past();
}

template <typename T>
void write_list(list<T>& list_to_write, ofstream& output){
    if(list_to_write.empty()){
        output << "[]";
        return;
    }

    output << "[";
    bool first = true;
    for(auto val : list_to_write) {
        if (!first) { output << ","; }
        else first = false;
        output << val;
    }
    output << "]";
}


void predict_trace(state_merger* m, ofstream& output, trace* tr){
    if(REVERSE_TRACES) tr->reverse();

    state_sequence.clear();
    score_sequence.clear();
    align_sequence.clear();
    ending_state = nullptr;
    ending_tail = nullptr;

    if(PREDICT_ALIGN) {
        align(m, tr->get_head(), true, 1.0);
    } else {
        predict_trace_update_sequences(m, tr->get_head());
    }

    output << rownr << "; " << "\"" << tr->to_string() << "\"";

    output << "; ";
    write_list(state_sequence, output);
    output << "; ";
    write_list(score_sequence, output);

    if(SLIDING_WINDOW) {
        auto score_it = score_sequence.begin();
        auto state_it = state_sequence.begin();
        auto align_it = align_sequence.begin();
        tail *tail_it = tr->get_head();
        while (score_it != score_sequence.end() && tail_it != nullptr && !tail_it->is_final()) {
            sw_individual_tail_score[tail_it] = *score_it;

            int tail_nr = tail_it->get_nr();
            if (sw_score_per_symbol.find(tail_nr) == sw_score_per_symbol.end())
                sw_score_per_symbol[tail_nr] = *score_it;
            else sw_score_per_symbol[tail_nr] += *score_it;
            if (*align_it) { tail_it = tail_it->future(); }
            ++state_it;
            ++score_it;
            ++align_it;
        }

        list<double> score_tail_sequence;
        list<int> tail_nr_sequence;

        tail_it = tr->get_head();
        while(tail_it != nullptr && !tail_it->is_final()){
            int tail_nr = tail_it->get_nr();
            tail_nr_sequence.push_back(tail_nr);
            if (sw_score_per_symbol.find(tail_nr) != sw_score_per_symbol.end())
                score_tail_sequence.push_back(sw_score_per_symbol[tail_nr]);
            tail_it = tail_it->future();
        }

        output << "; ";
        write_list(score_tail_sequence, output);

        double front_tail_score = score_tail_sequence.front();
        if(!score_tail_sequence.empty()) output << "; " << front_tail_score;
        else output << "; " << 0;

        list<int> row_nrs_front_tail;
        tail* front_tail = tr->get_head();
        tail* root_cause = front_tail;
        while(front_tail != nullptr){
            row_nrs_front_tail.push_back(front_tail->get_sequence());
                if(front_tail_score > 0 && sw_individual_tail_score[front_tail] > 1.5 * sw_individual_tail_score[root_cause]){
                    root_cause = front_tail;
                }
                if(front_tail_score < 0 && sw_individual_tail_score[front_tail] < 1.5 * sw_individual_tail_score[root_cause]){
                    root_cause = front_tail;
                }
            front_tail = front_tail->split_from;
        }
        output << "; " << "\"" << root_cause->to_string() << "\"";
        output << "; ";
        write_list(row_nrs_front_tail, output);
    }

    if(PREDICT_ALIGN){
        output << "; ";
        write_list(align_sequence, output);
        int nr_misaligned = 0;
        for(bool b : align_sequence){ if(!b) nr_misaligned++; }
        output << "; " << nr_misaligned;
    }

    if(PREDICT_TRACE){
        if(score_sequence.empty()){
            output << " 0; 0; 0;";
        } else {
            double sum = 0.0;
            for(auto val : score_sequence) sum += val;
            output << "; " << sum << "; " << (sum / (double)score_sequence.size());

            double min = 0.0;
            for(auto val : score_sequence) if(val < min) min = val;
            output << "; " << min;
        }
    }

    if(ending_state != nullptr){
        if(PREDICT_TYPE){
            output << "; " << inputdata_locator::get()->string_from_type(tr->get_type());
            output << "; " << ending_state->get_data()->predict_type_score(tr->get_head());

            int type_predict = ending_state->get_data()->predict_type(ending_tail);
            output << "; " << inputdata_locator::get()->string_from_type(type_predict);
            output << "; " << ending_state->get_data()->predict_type_score(type_predict);
        }

        if(PREDICT_SYMBOL) {
            if (ending_tail != nullptr) {
                output << "; " << inputdata_locator::get()->string_from_symbol(ending_tail->get_symbol());
                output << "; " << ending_state->get_data()->predict_symbol_score(ending_tail);
            } else output << "; 0; 0";

            int symbol_predict = ending_state->get_data()->predict_symbol(ending_tail);
            output << "; " << inputdata_locator::get()->string_from_symbol(symbol_predict);
            output << "; " << ending_state->get_data()->predict_symbol_score(symbol_predict);
        }

        if(PREDICT_DATA) {
            if (ending_tail != nullptr) {
                output << "; " << ending_tail->get_data();
                output << "; " << ending_state->get_data()->predict_data_score(ending_tail);
            } else output << "; 0; 0";

            string data_predict = ending_state->get_data()->predict_data(ending_tail);
            output << "; " << data_predict;
            output << "; " << ending_state->get_data()->predict_data_score(data_predict);
        }
    }
    else{
        if(PREDICT_TYPE){
            output << "; 0; 0; 0; 0";
        }
        if(PREDICT_SYMBOL){
            output << "; 0; 0; 0; 0";
        }
        if(PREDICT_DATA){
            output << "; 0; 0; 0; 0";
        }
    }
    output << endl;
}

void predict_csv(state_merger* m, istream& input, ofstream& output){
    inputdata* id = m->get_dat();
    rownr = -1;

    output << "row nr; abbadingo trace; state sequence; score sequence";
    if(SLIDING_WINDOW) output << "; score per sw tail; score first sw tail; root cause sw tail score; row nrs first sw tail";
    if(PREDICT_ALIGN) output << "; alignment; num misaligned";
    if(PREDICT_TRACE) output << "; sum scores; mean scores; min score";
    if(PREDICT_TYPE) output << "; trace type; type probability; predicted trace type; predicted type probability";
    if(PREDICT_SYMBOL) output << "; next trace symbol; next symbol probability; predicted symbol; predicted symbol probability";
    output << endl;

    //TODO fix
//    while(!input.eof()) {
//        rownr += 1;
//        trace* tr = id->read_csv_row(input);
//        if(tr == nullptr) continue;
//        if(!tr->get_end()->is_final()){
//            continue;
//        }
//        predict_trace(m, output, tr);
//        tail* tail_it = tr->get_head();
//        //cerr << "predicted " << tr->to_string() << " " << tail_it->get_nr() << " " << tail_it->tr->get_sequence() << endl;
//        for(int i = 0; i < SLIDING_WINDOW_STRIDE; i++){
//            tail* tail_to_delete = tail_it;
//            while(tail_to_delete->split_from != nullptr) tail_to_delete = tail_to_delete->split_from;
//            if(tail_to_delete->get_index() < SLIDING_WINDOW_SIZE - SLIDING_WINDOW_STRIDE) continue;
//            add_visits(m, tail_to_delete->tr);
//            //cerr << "deleting " << tail_to_delete->tr->to_string() << " " << tail_to_delete->get_nr() << " " << tail_to_delete->tr->get_sequence() << endl;
//            tail_to_delete->tr->erase();
//            tail_it = tail_it->future();
//        }
//    }
}

void predict(state_merger* m, istream& input, ofstream& output){
    output << "row nr; abbadingo trace; state sequence; score sequence";
    if(SLIDING_WINDOW) output << "; score per sw tail; score first sw tail; root cause sw tail score; row nrs first sw tail";
    if(PREDICT_ALIGN) output << "; alignment; num misaligned";
    if(PREDICT_TRACE) output << "; sum scores; mean scores; min score";
    if(PREDICT_TYPE) output << "; trace type; type probability; predicted trace type; predicted type probability";
    if(PREDICT_SYMBOL) output << "; next trace symbol; next symbol probability; predicted symbol; predicted symbol probability";
    output << endl;

    // TODO fix
//    rownr=-1;
//    inputdata* id = m->get_dat();
//    for(int i = 0; i < id->get_max_sequences(); ++i) {
//        rownr += 1;
//        trace* tr = mem_store::create_trace();
//        id->read_abbadingo_sequence(input, tr);
//        predict_trace(m, output, tr);
//        add_visits(m, tr);
//        tr->erase();
//    }
}