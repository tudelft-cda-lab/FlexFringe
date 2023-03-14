#include <queue>
#include <iostream>
#include "refinement.h"
#include "parameters.h"
#include "apta.h"
#include "input/inputdata.h"
#include "input/inputdatalocator.h"
#include "state_merger.h"

using namespace std;

refinement::refinement(){
    score = 0.0;
    red = nullptr;
    red_trace = nullptr;
    size = 0;
    refs = 0;

    time = 0;
}

merge_refinement::merge_refinement(state_merger* m, double s, apta_node* l, apta_node* r){
    red = l;
    red_trace = m->get_trace_from_state(l);
    red_trace->inc_refs();
    blue = r;
    blue_trace = m->get_trace_from_state(r);
    blue_trace->inc_refs();
    score = s;
    size = r->get_size();
    refs = 1;
    time = m->get_num_merges();
    if(RANDOMIZE_SCORES > 0.0) score = score - (score * RANDOMIZE_SCORES * random_double());
}

void merge_refinement::initialize(state_merger* m, double s, apta_node* l, apta_node* r){
    red = l;
    red_trace = m->get_trace_from_state(l);
    red_trace->inc_refs();
    blue = r;
    blue_trace = m->get_trace_from_state(r);
    blue_trace->inc_refs();
    score = s;
    size = r->get_size();
    refs = 1;
    time = m->get_num_merges();
    if(RANDOMIZE_SCORES > 0.0) score = score - (score * RANDOMIZE_SCORES * random_double());
}

split_refinement::split_refinement(state_merger* m, double s, apta_node* r, tail* t, int a){
    split_point = inputdata_locator::get()->access_tail(t);
    red = r;
    red_trace = m->get_trace_from_state(r);
    red_trace->inc_refs();
    score = s;
    attribute = a;
    size = r->get_size();
    refs = 1;
    time = m->get_num_merges();
    if(RANDOMIZE_SCORES > 0.0) score = score - (score * RANDOMIZE_SCORES * random_double());
}

void split_refinement::initialize(state_merger* m, double s, apta_node* r, tail* t, int a){
    split_point = t;
    red = r;
    red_trace = m->get_trace_from_state(r);
    red_trace->inc_refs();
    score = s;
    attribute = a;
    size = r->get_size();
    refs = 1;
    time = m->get_num_merges();
    if(RANDOMIZE_SCORES > 0.0) score = score - (score * RANDOMIZE_SCORES * random_double());
}

extend_refinement::extend_refinement(state_merger* m, apta_node* r){
    red = r;
    red_trace = m->get_trace_from_state(r);
    red_trace->inc_refs();
    score = EXTEND_SCORE;
    size = r->get_size();
    refs = 1;
    time = m->get_num_merges();
    if(RANDOMIZE_SCORES > 0.0) score = score - (score * RANDOMIZE_SCORES * random_double());
}

void extend_refinement::initialize(state_merger* m, apta_node* r){
    red = r;
    red_trace= m->get_trace_from_state(r);
    red_trace->inc_refs();
    score = 0.0;
    size = r->get_size();
    refs = 1;
    time = m->get_num_merges();
    if(RANDOMIZE_SCORES > 0.0) score = score - (score * RANDOMIZE_SCORES * random_double());
}

inline void refinement::print() const{
    cout << "score " << score << endl;
};

inline void refinement::print_json(iostream& output) const{
    output << "\t\t[\n";
    output << "score " << score << endl;
    output << "\t\t]\n";
};

inline void refinement::print_short() const{
    cout << score;
};

inline void refinement::doref(state_merger* m){
};
	
inline void refinement::undo(state_merger* m){
};

inline bool refinement::testref(state_merger* m){
    if(this->test_ref_structural(m)){
        return this->test_ref_consistency(m);
    }

    return false;
};

inline bool refinement::test_ref_structural(state_merger* m){
    return true;
};

inline bool refinement::test_ref_consistency(state_merger* m){
    return true;
};

inline void refinement::increfs(){
    ++refs;
};

inline void refinement::erase(){
};



inline void merge_refinement::print() const{
    if(STORE_ACCESS_STRINGS)
        cout << "merge( " << score << ", " << red_trace->to_string() << ", " << blue_trace->to_string() << " )" << endl;
    else
        cout << "merge( " << score << ", " << red->get_number() << ", " << blue->get_number() << " )" << endl;
};
	
inline void merge_refinement::print_short() const{
    cout << "m" << score;
};

inline void merge_refinement::print_json(iostream& output) const{
    output << "\t\t{\n";
    output << "\t\t\t\"type\" : \"merge\", " << endl;
    output << "\t\t\t\"red\" : " << red->get_number() << "," << endl;
    output << "\t\t\t\"blue\" : " << blue->get_number() << "," << endl;
    output << "\t\t\t\"score\" : " << score << endl;
    output << "\t\t}\n";
};

inline void merge_refinement::doref(state_merger* m){
    apta_node* left = red;
    apta_node* right = blue;
    if(STORE_ACCESS_STRINGS){
        left = m->get_state_from_trace(red_trace);
        right = m->get_state_from_trace(blue_trace);
    }
    if(!left->is_red()){
        /** this is a blueblue merge */
        m->extend(left);
        right->set_red(true);
    }
    m->perform_merge(left, right);
};
	
inline void merge_refinement::undo(state_merger* m){
    apta_node* left = red;
    apta_node* right = blue;
    if(STORE_ACCESS_STRINGS){
        left = m->get_state_from_trace(red_trace);
        right = m->get_state_from_trace(blue_trace);
    }
    m->undo_perform_merge(left, right);
    if(right->is_red()){
        /** this was a blueblue merge */
        right->set_red(false);
        m->undo_extend(left);
    }
};

/* inline bool merge_refinement::testref(state_merger* m){
    if(this->test_ref_structural(m)){
        return this->test_ref_consistency(m);
    }

    return false;
}; */

inline bool merge_refinement::test_ref_structural(state_merger* m){
    apta_node* left = red;
    apta_node* right = blue;
    if(STORE_ACCESS_STRINGS){
        left = m->get_state_from_trace(red_trace);
        right = m->get_state_from_trace(blue_trace);
    }

    if(left == right) return false;
    if((!left->is_red() && !left->get_source()->find()->is_red()) || right->is_red() || !right->get_source()->find()->is_red()) return false;
    if(left->rep() != 0 || right->rep() != 0) return false;

    return true;
};

inline bool merge_refinement::test_ref_consistency(state_merger* m){
    apta_node* left = red;
    apta_node* right = blue;
    if(STORE_ACCESS_STRINGS){
        left = m->get_state_from_trace(red_trace);
        right = m->get_state_from_trace(blue_trace);
    }

    refinement* ref = m->test_merge(left, right);
    if(ref != 0){
        score = ref->score;
        return true;
    }
    return false;
};

inline void merge_refinement::erase(){
    refs -= 1;
    if(refs == 0) mem_store::delete_merge_refinement(this);
};

inline void split_refinement::print() const{
    if(STORE_ACCESS_STRINGS)
        cout << "split( " << score << " q:" << red_trace->to_string() << " s:" << split_point->to_string() << " a:" << attribute << " )";
    else
        cout << "split( " << score << " q:" << red->get_number() << " s:" << split_point->to_string() << " a:" << attribute << " )";
};
	
inline void split_refinement::print_short() const{
    cout << "s" << score;
};

inline void split_refinement::print_json(iostream& output) const{
    output << "\t\t{\n";
    output << "\t\t\t\"type\" : \"split\", " << endl;
    output << "\t\t\t\"red\" : " << red->get_number() << "," << endl;
    output << "\t\t\t\"point\" : " << split_point->to_string() << "," << endl;
    output << "\t\t\t\"attribute\" : " << attribute << "," << endl;
    output << "\t\t\t\"score\" : " << score << endl;
    output << "\t\t}\n";
};

inline void split_refinement::doref(state_merger* m){
    apta_node* right = red;
    if(STORE_ACCESS_STRINGS){
        right = m->get_state_from_trace(red_trace);
    }
    m->perform_split(right, split_point, attribute);
};
	
inline void split_refinement::undo(state_merger* m){
    apta_node* right = red;
    if(STORE_ACCESS_STRINGS){
        right = m->get_state_from_trace(red_trace);
    }
    m->undo_perform_split(right, split_point, attribute);
};

inline bool split_refinement::testref(state_merger* m){
    apta_node* right = red;
    if(STORE_ACCESS_STRINGS){
        right = m->get_state_from_trace(red_trace);
    }
    if(!right->is_red()) return false;
    if(right->guard(split_point) == 0) return false;
    if(right->guard(split_point)->get_target() == 0) return false;
    if(right->guard(split_point)->get_target()->rep() != 0) return false;
    if(right->guard(split_point)->get_target()->is_red()) return false;
    refinement* ref = m->test_split(right, split_point, attribute);
    if(ref != 0){
        score = ref->score;
        return true;
    }
    return false;
};

// TODO: I am not sure if we want those two to be 
inline bool split_refinement::test_ref_structural(state_merger* m){
    throw runtime_error("test_ref_structural() not yet implemented for split_refinement. Aborting");
    //return refinement::test_ref_structural(m);
}

inline bool split_refinement::test_ref_consistency(state_merger* m){
    throw runtime_error("test_ref_consistency() not yet implemented for split_refinement. Aborting");
    //return refinement::test_ref_consistency(m);
}

inline void split_refinement::erase(){
    refs -= 1;
    if(refs == 0) mem_store::delete_split_refinement(this);
};

inline void extend_refinement::print() const{
    if(STORE_ACCESS_STRINGS)
        cout << "extend( " << score << " " << red_trace->to_string() << " )" << endl;
    else
        cout << "extend( " << score << " " << red->get_number() << " )" << endl;
};
	
inline void extend_refinement::print_short() const{
    cout << "x" << size;
};

inline void extend_refinement::print_json(iostream& output) const{
    output << "\t\t{\n";
    output << "\t\t\t\"type\" : \"extend\", " << endl;
    output << "\t\t\t\"red\" : " << red->get_number() << "," << endl;
    output << "\t\t\t\"score\" : " << score << endl;
    output << "\t\t}\n";
};

inline void extend_refinement::doref(state_merger* m){
    apta_node* right = red;
    if(STORE_ACCESS_STRINGS){
        right = m->get_state_from_trace(red_trace);
    }
    m->extend(right);
};
	
inline void extend_refinement::undo(state_merger* m){
    apta_node* right = red;
    if(STORE_ACCESS_STRINGS){
        right = m->get_state_from_trace(red_trace);
    }
    m->undo_extend(right);
};

inline bool extend_refinement::testref(state_merger* m){
    apta_node* right = red;
    if(STORE_ACCESS_STRINGS){
        right = m->get_state_from_trace(red_trace);
    }
    if(right->is_red() || !right->get_source()->find()->is_red()) return false;
    if(right->rep() != 0) return false;
    return true;
};

inline bool extend_refinement::test_ref_structural(state_merger* m){
    return extend_refinement::testref(m);
};

inline bool extend_refinement::test_ref_consistency(state_merger* m){
    return refinement::test_ref_consistency(m);
};

inline void extend_refinement::erase(){
    refs -= 1;
    if(refs == 0) mem_store::delete_extend_refinement(this);
};

void refinement::print_refinement_list_json(iostream& output, refinement_list* list){
    output << "[\n";

    refinement_list::iterator it = list->begin();
    while(it != list->end()){
        (*it)->print_json(output);
        it++;
        if(it != list->end())
            output << ",\n";
    }

    output << "]\n";
};

refinement_type refinement::type(){
    throw runtime_error("type() function of refinement base class should not be called.");
    return refinement_type::base_rf_type;
}
refinement_type split_refinement::type(){
    return refinement_type::split_rf_type;
}
refinement_type merge_refinement::type(){
    return refinement_type::merge_rf_type;
}
refinement_type extend_refinement::type(){
    return refinement_type::extend_rf_type;
}

inline int refinement::get_time(){
    return time;
}

