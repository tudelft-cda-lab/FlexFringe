/// @file dfasat.cpp
/// @brief Reduction for the SAT solver, loop for the combined heuristic-SAT mode
/// @author Sicco Verwer

#include <stdio.h>
#include <stdlib.h>
#include "dfasat.h"
#include <sstream>
#include <errno.h>
#include <string.h>
#include <set>
#include <vector>
#include <ctime>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "parameters.h"
#include "conflict_graph.h"
#include "input/inputdatalocator.h"

void dfasat::reset_literals(bool init){
    int v, i, j, a, t; // TODO: aren't there better names for those?

    literal_counter = 1;
    for(v = 0; v < num_states; ++v)
        for(i = 0; i < dfa_size; ++i)
            if(init || x[v][i] > 0) x[v][i] = literal_counter++;
    
    for(a = 0; a < alphabet_size; ++a)
        for(i = 0; i < dfa_size; ++i)
            for(j = 0; j < dfa_size; ++j)
                if(init || y[a][i][j] > 0) y[a][i][j] = literal_counter++;

    for(a = 0; a < alphabet_size; ++a)
        for(i = 0; i < dfa_size; ++i)
            for(j = 0; j < sinks_size; ++j)
                if(init || sy[a][i][j] > 0) sy[a][i][j] = literal_counter++;

    for(i = 0; i < num_states; ++i)
        if(init || sp[i] > 0) sp[i] = literal_counter++;
    
    for(i = 0; i < dfa_size; ++i)
        for(t = 0; t < num_types; ++t)
            if(init || z[i][t] > 0) z[i][t] = literal_counter++;

    for(i = 0; i < dfa_size; ++i)
        for(j = 0; j < new_states; ++j)
            if(init || yt[i][j] > 0) yt[i][j] = literal_counter++;
    
    for(i = 0; i < dfa_size; ++i)
        for(j = 0; j < new_states; ++j)
            if(init || yp[i][j] > 0) yp[i][j] = literal_counter++;
    
    for(a = 0; a < alphabet_size; ++a)
        for(i = 0; i < new_states; ++i)
            if(init || ya[a][i] > 0) ya[a][i] = literal_counter++;
}

/**
 * @brief Create the literals (boolean expressions) for expressing constraints. 
 * For more information see e.g. Exact DFA Identification Using SAT Solvers, Heule and Verwer
 * 
 */
void dfasat::create_literals(){
    int v, a, i;
    //X(STATE,COLOR)
    x = (int**) malloc( sizeof(int*) * num_states);
    for(v = 0; v < num_states; v++ )
        x[ v ] = (int*) malloc( sizeof(int) * dfa_size);
    
    //Y(LABEL,COLOR,COLOR)
    y = (int***) malloc( sizeof(int**) * alphabet_size);
    for(a = 0; a < alphabet_size; ++a){
        y[ a ] = (int**) malloc( sizeof(int*) * dfa_size);
        for(i = 0; i < dfa_size; ++i)
            y[ a ][ i ]  = (int*) malloc( sizeof(int) * dfa_size);
    }

    //SY(LABEL,COLOR,SINK)
    sy = (int***) malloc( sizeof(int**) * alphabet_size);
    for(a = 0; a < alphabet_size; ++a){
        sy[ a ] = (int**) malloc( sizeof(int*) * dfa_size);
        for(i = 0; i < dfa_size; ++i)
            sy[ a ][ i ]  = (int*) malloc( sizeof(int) * sinks_size);
    }

    //SP(STATE)
    sp = (int*) malloc( sizeof(int) * num_states);
    
    //Z(COLOR)
    z = (int**) malloc( sizeof(int*) * dfa_size);
    for(i = 0; i < dfa_size; ++i){
        z[ i ] = (int*) malloc( sizeof(int) * num_types);
    }

    //YT(COLOR,COLOR)
    yt = (int**) malloc( sizeof(int*) * dfa_size);
    for(i = 0; i < dfa_size; ++i)
        yt[ i ]  = (int*) malloc( sizeof(int) * new_states);
    
    //YP(COLOR,COLOR)
    yp = (int**) malloc( sizeof(int*) * dfa_size);
    for(i = 0; i < dfa_size; ++i)
        yp[ i ]  = (int*) malloc( sizeof(int) * new_states);
    
    //YA(LABEL,COLOR)
    ya = (int**) malloc( sizeof(int*) * alphabet_size);
    for(a = 0; a < alphabet_size; ++a)
        ya[ a ]  = (int*) malloc( sizeof(int) * new_states);
    
    // reset literal values
    reset_literals(true);
}

/**
 * @brief Delete the literals again. 
 * 
 */
void dfasat::delete_literals(){
    int v, a, i;
    for(v = 0; v < num_states; v++ )
        free(x[ v ]);
    free(x);
    for(a = 0; a < alphabet_size; ++a){
        for(i = 0; i < dfa_size; ++i)
            free(y[ a ][ i ]);
        free(y[ a ]);
    }
    free(y);
    for(a = 0; a < alphabet_size; ++a){
        for(i = 0; i < sinks_size; ++i)
            free(sy[ a ][ i ]);
        free(sy[ a ]);
    }
    free(sy);
    for(i = 0; i < dfa_size; ++i)
        free(z[ i ]);
    free(z);
    free(sp);
    for(i = 0; i < dfa_size; ++i)
        free(yt[ i ]);
    free(yt);
    for(i = 0; i < dfa_size; ++i)
        free(yp[ i ]);
    free(yp);
    for(a = 0; a < alphabet_size; ++a)
        free(ya[ a ]);
    free(ya);
}

/**
 * @brief Print clauses without eliminated literals, -2 = false, -1 = true
 * 
 * @param v1 
 * @param l1 
 * @param v2 
 * @param l2 
 * @param v3 
 * @param l3 
 * @param v4 
 * @param l4 
 * @return int 
 */
int dfasat::print_clause(bool v1, int l1, bool v2, int l2, bool v3, int l3, bool v4, int l4){
    if(v1 && l1 == -1) return 0;
    if(!v1 && l1 == -2) return 0;
    if(v2  && l2 == -1) return 0;
    if(!v2 && l2 == -2) return 0;
    if(v3 && l3 == -1) return 0;
    if(!v3 && l3 == -2) return 0;
    if(v4 && l4 == -1) return 0;
    if(!v4 && l4 == -2) return 0;
    
    if(computing_header) return 1;
    
    /*
    if(v1 == true  && l1 != -2) fprintf(sat_stream, "%i ", l1);
    if(v1 == false && l1 != -1) fprintf(sat_stream, "-%i ", l1);
    if(v2 == true  && l2 != -2) fprintf(sat_stream, "%i ", l2);
    if(v2 == false && l2 != -1) fprintf(sat_stream, "-%i ", l2);
    if(v3 == true  && l3 != -2) fprintf(sat_stream, "%i ", l3);
    if(v3 == false && l3 != -1) fprintf(sat_stream, "-%i ", l3);
    if(v4 == true  && l4 != -2) fprintf(sat_stream, "%i ", l4);
    if(v4 == false && l4 != -1) fprintf(sat_stream, "-%i ", l4);
    
    fprintf(sat_stream, " 0\n");
    */

    if(v1 == true  && l1 != -2) sat_stream << l1 << " ";
    if(v1 == false && l1 != -1) sat_stream << -l1 << " ";
    if(v2 == true  && l2 != -2) sat_stream << l2 << " ";
    if(v2 == false && l2 != -1) sat_stream << -l2 << " ";
    if(v3 == true  && l3 != -2) sat_stream << l3 << " ";
    if(v3 == false && l3 != -1) sat_stream << -l3 << " ";
    if(v4 == true  && l4 != -2) sat_stream << l4 << " ";
    if(v4 == false && l4 != -1) sat_stream << -l4 << " ";

    sat_stream << " 0\n";

    return 1;
}

/**
 * @brief TODO
 * 
 * @param v1 
 * @param l1 
 * @param v2 
 * @param l2 
 * @param v3 
 * @param l3 
 * @return int 
 */
int dfasat::print_clause(bool v1, int l1, bool v2, int l2, bool v3, int l3){
    if(v1 && l1 == -1) return 0;
    if(!v1 && l1 == -2) return 0;
    if(v2  && l2 == -1) return 0;
    if(!v2 && l2 == -2) return 0;
    if(v3 && l3 == -1) return 0;
    if(!v3 && l3 == -2) return 0;
    
    if(computing_header) return 1;
    
    /*
    if(v1 == true  && l1 != -2) fprintf(sat_stream, "%i ", l1);
    if(v1 == false && l1 != -1) fprintf(sat_stream, "-%i ", l1);
    if(v2 == true  && l2 != -2) fprintf(sat_stream, "%i ", l2);
    if(v2 == false && l2 != -1) fprintf(sat_stream, "-%i ", l2);
    if(v3 == true  && l3 != -2) fprintf(sat_stream, "%i ", l3);
    if(v3 == false && l3 != -1) fprintf(sat_stream, "-%i ", l3);
    
    fprintf(sat_stream, " 0\n");
    */

    if(v1 == true  && l1 != -2) sat_stream << l1 << " ";
    if(v1 == false && l1 != -1) sat_stream << -l1 << " ";
    if(v2 == true  && l2 != -2) sat_stream << l2 << " ";
    if(v2 == false && l2 != -1) sat_stream << -l2 << " ";
    if(v3 == true  && l3 != -2) sat_stream << l3 << " ";
    if(v3 == false && l3 != -1) sat_stream << -l3 << " ";

    sat_stream << " 0\n";

    return 1;
}

/**
 * @brief TODO
 * 
 * @param v1 
 * @param l1 
 * @param v2 
 * @param l2 
 * @return int 
 */
int dfasat::print_clause(bool v1, int l1, bool v2, int l2){
    if(v1 && l1 == -1) return 0;
    if(!v1 && l1 == -2) return 0;
    if(v2  && l2 == -1) return 0;
    if(!v2 && l2 == -2) return 0;

    if(computing_header) return 1;
    
    /*
    if(v1 == true  && l1 != -2) fprintf(sat_stream, "%i ", l1);
    if(v1 == false && l1 != -1) fprintf(sat_stream, "-%i ", l1);
    if(v2 == true  && l2 != -2) fprintf(sat_stream, "%i ", l2);
    if(v2 == false && l2 != -1) fprintf(sat_stream, "-%i ", l2);
    
    fprintf(sat_stream, " 0\n");
     */

    if(v1 == true  && l1 != -2) sat_stream << l1 << " ";
    if(v1 == false && l1 != -1) sat_stream << -l1 << " ";
    if(v2 == true  && l2 != -2) sat_stream << l2 << " ";
    if(v2 == false && l2 != -1) sat_stream << -l2 << " ";

    sat_stream << " 0\n";

    return 1;
}

bool dfasat::always_true(int number, bool flag){
    if(number == -1 && flag == true)  return true;
    if(number == -2 && flag == false) return true;
    return false;
}

void dfasat::print_lit(int number, bool flag){
    if(computing_header) return;
    if(number < 0) return;
    
    /*
    if(flag == true) fprintf(sat_stream, "%i ", number);
    else fprintf(sat_stream, "-%i ", number);
    */
    if(flag == true) sat_stream << number << " ";
    else sat_stream << -number << " ";
}

void dfasat::print_clause_end(){
    if(computing_header) return;
    //fprintf(sat_stream, " 0\n");
    sat_stream << " 0\n";
}

/** fix values for red states -2 = false, -1 = true */
void dfasat::fix_red_values(){
    for(state_set::iterator it = red_states->begin();it != red_states->end();++it){
        apta_node* node = *it;

        int nr = state_number[node];
        int cr = state_colour[node];

        for(int i = 0; i < dfa_size; ++i) x[nr][i] = -2;
        sp[nr] = -2;
        x[nr][cr] = -1;
        
        for(int label = 0; label < alphabet_size; ++label){
            apta_node* target = node->get_child(label);
            if(target != nullptr && target->is_red() == true){
                int tcr = state_colour[target];

                for(int i = 0; i < dfa_size; ++i) y[label][cr][i] = -2;
                for(int i = 0; i < sinks_size; ++i) sy[label][cr][i] = -2;
                y[label][cr][tcr] = -1;
            }
        }
        
        //if(node->pos_final() != 0) z[node->colour] = -1;
        //if(node->neg_final() != 0) z[node->colour] = -2;

        graph_node* gn = ag->get_node(node);
        for(type_set::iterator it = gn->pos_types.begin(); it != gn->pos_types.end(); ++it){
            z[cr][*it] = -1;
        }
        for(type_set::iterator it = gn->neg_types.begin(); it != gn->neg_types.end(); ++it){
            z[cr][*it] = -2;
        }

        //if(node->type == 1) z[node->colour] = -1;
        //if(node->type != 1) z[node->colour] = -2;
    }
}

void dfasat::fix_sink_values(){
    for(state_set::iterator it = red_states->begin(); it != red_states->end(); ++it){
        apta_node* node = *it;
        int cr = state_colour[node];

        for(int label = 0; label < alphabet_size; ++label){
            apta_node* target = node->get_child(label);

            if(MERGE_SINKS_PRESOLVE && target != 0 && sink_states->find(target) != sink_states->end()){
                for(int i = 0; i < dfa_size; ++i) y[label][cr][i] = -2;
                for(int i = 0; i < sinks_size; ++i) sy[label][cr][i] = -2;
                sy[label][cr][merger->sink_type(target)] = -1;
            } else if(TARGET_REJECTING && target == 0){
                for(int i = 0; i < dfa_size; ++i) y[label][cr][i] = -2;
                for(int i = 0; i < sinks_size; ++i) sy[label][cr][i] = -2;
                sy[label][cr][0] = -1;
            }
        }
    }
}

/**
 * @brief Erase possible colors due to symmetry reduction.
 * Should be compatible with BFS symmetry breaking, unchecked
 * 
 * @return int 
 */
int dfasat::set_symmetry(){
    int num = 0;
    int max_value = new_init;
    for(state_set::iterator it = red_states->begin(); it != red_states->end(); ++it){
        if(max_value + 1>= dfa_size)
            break;
        
        apta_node* node = *it;
        for(int a = 0; a < alphabet_size; ++a){

            apta_node* child = node->get_child(a);
            if(child != 0 && child->is_red()){
                int nr = state_number[child];

                if(MERGE_SINKS_PRESOLVE && sink_states->find(child) != sink_states->end())
                    continue;
                
                for(int i = max_value + 1; i < dfa_size; ++i){
                    x[nr][i] = -2;
                }
                max_value++;
            }
        }
    }
    return num;
}

int dfasat::print_symmetry(){
    int num = 0;
    for(int i = 0; i < dfa_size; ++i){
        for(int k = 0; k < new_states; ++k){
            for(int j = 0; j < i; ++j){
                for(int l = k + 1; l < new_states; l++){
                    num += print_clause(false, yp[i][k], false, yp[j][l]);
                }
            }
        }
    }
    for(int i = 0; i < dfa_size; ++i){
        for(int k = 0; k < new_states; ++k){
            for(int l = k + 1; l < new_states; l++){
                for(int a = 0; a < alphabet_size; ++a){
                    for(int b = 0; b < a; ++b){
                        num += print_clause(false, yp[i][k], false, yp[i][l], false, ya[a][k], false, ya[b][l]);
                    }
                }
            }
        }
    }
    return num;
}

/**
 * @brief Eliminate literals for merges that conflict with the red states.
 * 
 */
void dfasat::erase_red_conflict_colours(){
    for(state_set::iterator it = red_states->begin(); it != red_states->end(); ++it){
        apta_node* left = *it;
        int left_cl = state_colour[left];
        for(state_set::iterator it2 = non_red_states->begin(); it2 != non_red_states->end(); ++it2){
            apta_node* right = *it2;
            int right_nr = state_number[right];

            //graph_node* l = ag->get_node(left);
            //graph_node* r = ag->get_node(right);

            //if(l->neighbors.find(r) != l->neighbors.end() || l->type_consistent(r) == false || r->type_consistent(l) == false){
            //    x[right->satnumber][left->colour] = -2;
            //}
            if(merger->test_merge(left,right) == 0) x[right_nr][left_cl] = -2;
            //if(merger.test_local_merge(left,right) == -1) x[right->satnumber][left->colour] = -2;
            //if(right->pos_paths() != 0 || right->pos_final() != 0) x[right->satnumber][0] = -2;
            //if(right->neg_paths() != 0 || right->neg_final() != 0) x[right->satnumber][1] = -2;
        }
    }
}

void erase_sink_conflict_colours(){
    /*for(int i = 0; i < num_sink_types; ++ i){
        for(state_set::iterator it2 = non_red_states->begin(); it2 != non_red_states->end(); ++it2){
            apta_node* right = *it2;
            if(merger.sink_consistent(right,i) == -1) x[right->satnumber][left->colour] = -2;
        }
    }*/
}

/**
 * @brief Print the at least one en at most one clauses for x
 * 
 * @return int TODO
 */
int dfasat::print_colours(){
    int num = 0;
    bool altr = false;
    // at least one
    for(state_set::iterator it = non_red_states->begin(); it != non_red_states->end(); ++it){
        apta_node* node = *it;
        int nr = state_number[node];

        altr = always_true(sp[nr], true);
        for(int k = 0; k < dfa_size; ++k){
            if(altr) break;
            altr = always_true(x[nr][k], true);
        }
        if(altr == false){
            for(int k = 0; k < dfa_size; ++k)
                print_lit(x[nr][k], true);
            //print_lit(sp[nr], true);
            print_clause_end();
            num += 1;
        }
    }
    // at most one
    for(state_set::iterator it = non_red_states->begin(); it != non_red_states->end(); ++it){
        apta_node* node = *it;
        int nr = state_number[node];

        for(int a = 0; a < dfa_size; ++a)
            for(int b = a+1; b < dfa_size; ++b)
                num += print_clause(false, x[nr][a], false, x[nr][b]);
        //for(int a = 0; a < dfa_size; ++a)
            //num += print_clause(false, x[nr][a], false, sp[nr]);
    }
    return num;
}

/**
 * @brief Print clauses restricting two unmergable states to have the same color.
 * Excludes pairs of states that are covered by the z literals.
 * 
 * @return int 
 */
int dfasat::print_conflicts(){
    int num = 0;
    for(state_set::iterator it = non_red_states->begin(); it != non_red_states->end(); ++it){
        apta_node* left = *it;
        state_set::iterator it2 = it;
        int left_nr = state_number[left];
        ++it2;
        while(it2 != non_red_states->end()){
            apta_node* right = *it2;
            int right_nr = state_number[right];
            ++it2;
            //if(left->pos_final() != 0 && right->neg_final() != 0) continue;
            //if(left->neg_final() != 0 && right->pos_final() != 0) continue;
            //if(left->type == 1 && right->type != 1) continue;
            //if(left->type != 1 && right->type == 1) continue;

            //graph_node* l = ag->get_node(left);
            //graph_node* r = ag->get_node(right);

            /*if(l->neighbors.find(r) != l->neighbors.end() || l->type_consistent(r) == false){
                for(int k = 0; k < dfa_size; ++k)
                    num += print_clause(false, x[left->satnumber][k], false, x[right->satnumber][k]);
            }*/

            if(merger->test_merge(left, right) == 0){
                //cerr << left << " and " << right << " cannot have the same colour" << endl;
                for(int k = 0; k < dfa_size; ++k)
                    num += print_clause(false, x[left_nr][k], false, x[right_nr][k]);
            }

            /*if(merger.test_local_merge(left, right) == -1){
                for(int k = 0; k < dfa_size; ++k)
                    num += print_clause(false, x[left->satnumber][k], false, x[right->satnumber][k]);
            }*/
        }
    }
    return num;
}

/**
 * @brief Print the clauses for z literals.
 * 
 * @return int 
 */
int dfasat::print_accept(){
    int num = 0;
    for(state_set::iterator it = non_red_states->begin(); it != non_red_states->end(); ++it){
        apta_node* node = *it;
        int nr = state_number[node];
        graph_node* gn = ag->get_node(node);

        for(int k = 0; k < dfa_size; ++k){
            //if(node->pos_final() != 0) num += print_clause(false, x[node->satnumber][k], true, z[k]);
            //if(node->neg_final() != 0) num += print_clause(false, x[node->satnumber][k], false, z[k]);

            for(type_set::iterator it = gn->pos_types.begin(); it != gn->pos_types.end(); ++it){
                num += print_clause(false, x[nr][k], true, z[k][*it]);
            }
            for(type_set::iterator it = gn->neg_types.begin(); it != gn->neg_types.end(); ++it){
                num += print_clause(false, x[nr][k], false, z[k][*it]);
            }

            //if(node->type == 1) num += print_clause(false, x[nr][k], true, z[k]);
            //if(node->type != 1) num += print_clause(false, x[nr][k], false, z[k]);
        }
    }
    return num;
}

/**
 * @brief Print the clauses for y literals.
 * 
 * @return int 
 */
int dfasat::print_transitions(){
    int num = 0;
    for(int a = 0; a < alphabet_size; ++a)
        for(int i = 0; i < dfa_size; ++i)
            for(int j = 0; j < dfa_size; ++j)
                for(int h = 0; h < j; ++h)
                    num += print_clause(false, y[a][i][h], false, y[a][i][j]);
    return num;
}

int dfasat::print_sink_transitions(){
    int num = 0;
    for(int a = 0; a < alphabet_size; ++a)
        for(int i = 0; i < dfa_size; ++i)
            for(int j = 0; j < sinks_size; ++j)
                for(int h = 0; h < dfa_size; ++h)
                    num += print_clause(false, y[a][i][h], false, sy[a][i][j]);
    for(int a = 0; a < alphabet_size; ++a)
        for(int i = 0; i < dfa_size; ++i)
            for(int j = 0; j < sinks_size; ++j)
                for(int h = 0; h < j; ++h)
                    num += print_clause(false, sy[a][i][h], false, sy[a][i][j]);
    return num;
}

/* print transitions for any label yt */
int dfasat::print_t_transitions(){
    int num = 0;
    for(int i = 0; i < dfa_size; ++i)
        for(int j = 0; j < new_states; ++j)
            for(int a = 0; a < alphabet_size; ++a)
                num += print_clause(false, y[a][i][new_init+j], true, yt[i][j]);
    
    for(int i = 0; i < dfa_size; ++i){
        for(int j = 0; j < new_states; ++j){
            bool altr = false;
            for(int a = 0; a < alphabet_size; ++a)
                if(y[a][i][new_init+j] == -1) altr = true;
            if(!altr){
                if(!computing_header){
                    print_lit(yt[i][j], false);
                    for(int a = 0; a < alphabet_size; ++a){
                        print_lit(y[a][i][new_init+j], true);
                    }
                    print_clause_end();
                }
                num++;
            }
        }
    }
    
    return num;
}

/* print BFS tree transitions */
int dfasat::print_p_transitions(){
    int num = 0;
    for(int i = 0; i < dfa_size; ++i){
        for(int j = 0; j < new_states; ++j){
            for(int k = 0; k < i; ++k){
                num += print_clause(false, yp[i][j], false, yt[k][j]);
            }
            num += print_clause(false, yp[i][j], true, yt[i][j]);
        }
    }
    for(int i = 0; i < new_states; ++i){
        bool altr = false;
        for(int j = 0; j < new_init+i; ++j)
            if(yp[j][i] == -1) altr = true;
        if(!altr){
            if(!computing_header){
                for(int j = 0; j < new_init+i; ++j){
                    print_lit(yp[j][i], true);
                }
                print_clause_end();
            }
            num++;
        }
    }
    return num;
}

/* print BFS tree labels */
int dfasat::print_a_transitions(){
    int num = 0;
    for(int i = 0; i < new_states; ++i){
        for(int a = 0; a < alphabet_size; ++a){
            for(int j = 0; j < dfa_size; ++j){
                for(int b = 0; b < a; ++b){
                    num += print_clause(false, ya[a][i], false, yp[j][i], false, y[b][j][new_init+i]);
                }
                num += print_clause(false, ya[a][i], false, yp[j][i], true, y[a][j][new_init+i]);
            }
        }
    }
    for(int i = 0; i < new_states; ++i){
        bool altr = false;
        for(int a = 0; a < alphabet_size; ++a){
            if(ya[a][i] == -1) altr = true;
        }
        if(!altr){
            if(!computing_header){
                for(int a = 0; a < alphabet_size; ++a){
                    print_lit(ya[a][i], true);
                }
                print_clause_end();
            }
            num++;
        }
    }
    return num;
}

/* print de clauses voor y literals */
int dfasat::print_forcing_transitions(){
    int num = 0;
    bool altr = false;
    for (int label = 0; label < alphabet_size; ++label) {
        state_set label_states;
        for (state_set::iterator it = red_states->begin(); it != red_states->end(); ++it) {
            apta_node* source = *it;
            if(source->get_child(label) != 0) label_states.insert(source);
        }
        for (state_set::iterator it = non_red_states->begin(); it != non_red_states->end(); ++it) {
            apta_node* source = *it;
            if(source->get_child(label) != 0) label_states.insert(source);
        }
        
        for(int i = 0; i < dfa_size; ++i){
            for(int j = 0; j < dfa_size; ++j){
                altr = always_true(y[label][i][j], false);
                for (state_set::iterator it = label_states.begin(); it != label_states.end(); ++it) {
                    if(altr) break;
                    apta_node* source = *it;
                    int nr = state_number[source];
                    altr = always_true(x[nr][i], true);
                }
                if(altr == false){
                    for (state_set::iterator it = label_states.begin(); it != label_states.end(); ++it) {
                        apta_node* source = *it;
                        int nr = state_number[source];
                        print_lit(x[nr][i], true);
                    }
                    print_lit(y[label][i][j], false);
                    print_clause_end();
                    num += 1;
                }
            }
        }
        
        for(int i = 0; i < dfa_size; ++i){
            for(int j = 0; j < sinks_size; ++j){
                altr = always_true(sy[label][i][j], false);
                for (state_set::iterator it = label_states.begin(); it != label_states.end(); ++it) {
                    if(altr) break;
                    apta_node* source = *it;
                    int nr = state_number[source];
                    altr = always_true(x[nr][i], true);
                }
                if(altr == false){
                    for (state_set::iterator it = label_states.begin(); it != label_states.end(); ++it) {
                        apta_node* source = *it;
                        int nr = state_number[source];
                        print_lit(x[nr][i], true);
                    }
                    print_lit(sy[label][i][j], false);
                    print_clause_end();
                    num += 1;
                }
            }
        }
    }
    return num;
}

/* print de determinization constraint */
int dfasat::print_paths(){
    int num = 0;
    for (state_set::iterator it = red_states->begin(); it != red_states->end(); ++it) {
        apta_node* source = *it;
        int nr = state_number[source];
        for (int label = 0; label < alphabet_size; ++label) {
            apta_node* target = source->get_child(label);
            if (target != 0 && sink_states->find(target) == sink_states->end()) {
                int tnr = state_number[target];
                for (int i = 0; i < dfa_size; ++i)
                    for (int j = 0; j < dfa_size; ++j)
                        num += print_clause(true, y[label][i][j], false, x[nr][i], false, x[tnr][j]);
            }
        }
    }
    for (state_set::iterator it = non_red_states->begin(); it != non_red_states->end(); ++it) {
        apta_node* source = *it;
        int nr = state_number[source];
        for (int label = 0; label < alphabet_size; ++label) {
            apta_node* target = source->get_child(label);
            if (target != 0) {
                int tnr = state_number[target];
                for (int i = 0; i < dfa_size; ++i)
                    for (int j = 0; j < dfa_size; ++j)
                        num += print_clause(true, y[label][i][j], false, x[nr][i], false, x[tnr][j]);
            }
        }
    }
    return num;
}

/* print sink paths */
int dfasat::print_sink_paths(){
    int num = 0;
    bool altr = false;
    for (state_set::iterator it = red_states->begin(); it != red_states->end(); ++it) {
        apta_node* source = *it;
        int nr = state_number[source];
        for (int label = 0; label < alphabet_size; ++label) {
            apta_node* target = source->get_child(label);
            if (target != 0 && sink_states->find(target) == sink_states->end()) {
                int tnr = state_number[target];
                for (int i = 0; i < dfa_size; ++i)
                    for (int j = 0; j < sinks_size; ++j)
                        if(merger->sink_consistent(target, j) == false)
                            num += print_clause(false, sy[label][i][j], false, x[nr][i]);
                
                for (int i = 0; i < dfa_size; ++i){
                    altr = always_true(x[nr][i], false);
                    if(!altr) altr = always_true(sp[tnr], false);
                    for(int j = 0; j < sinks_size; ++j){
                        if(altr) break;
                        if(merger->sink_consistent(target, j) == true) altr = always_true(sy[label][i][j], true);
                    }
                    
                    if(altr == false){
                        for(int j = 0; j < sinks_size; ++j)
                            if(merger->sink_consistent(target, j) == true) print_lit(sy[label][i][j], true);
                        print_lit(x[nr][i], false);
                        print_lit(sp[tnr], false);
                        print_clause_end();
                        num += 1;
                    }
                }
                num += print_clause(false, sp[nr], true, sp[tnr]);
            }
        }
    }
    for (state_set::iterator it = non_red_states->begin(); it != non_red_states->end(); ++it) {
        apta_node* source = *it;
        int nr = state_number[source];
        for (int label = 0; label < alphabet_size; ++label) {
            apta_node* target = source->get_child(label);
            if (target != 0) {
                int tnr = state_number[target];
                for (int i = 0; i < dfa_size; ++i)
                    for (int j = 0; j < sinks_size; ++j)
                        if(merger->sink_consistent(target, j) == false)
                            num += print_clause(false, sy[label][i][j], false, x[nr][i]);
                for (int i = 0; i < dfa_size; ++i){
                    altr = always_true(x[nr][i], false);
                    if(!altr) altr = always_true(sp[tnr], false);
                    for(int j = 0; j < sinks_size; ++j){
                        if(altr) break;
                        if(merger->sink_consistent(target, j) == true) altr = always_true(sy[label][i][j], true);
                    }
                    
                    if(altr == false){
                        for(int j = 0; j < sinks_size; ++j)
                            if(merger->sink_consistent(target, j) == true) print_lit(sy[label][i][j], true);
                        print_lit(x[nr][i], false);
                        print_lit(sp[tnr], false);
                        print_clause_end();
                        num += 1;
                    }
                }
                num += print_clause(false, sp[nr], true, sp[tnr]);
            }
        }
    }
    return num;
}

/* output result to dot */
void dfasat::print_dot_output(const char* dot_output){
    FILE* output = fopen(dot_output, "w");
    apta* aut = merger->get_aut();
    int i,a,j;
    
    fprintf(output,"digraph DFA {\n");
    fprintf(output,"\t\tI -> %i;\n", state_number[aut->get_root()->find()]);
    
    //set<int>::iterator it = trueliterals.begin();
    /*
    for(v = 0; v < num_states; ++v)
        for(i = 0; i < dfa_size; ++i)
            if(x[v][i] == *it) ++it;
    */

    for(a = 0; a < alphabet_size; ++a){
        for(i = 0; i < dfa_size; ++i){
            for(j = 0; j < dfa_size; ++j) {
                if(trueliterals.find(y[a][i][j]) != trueliterals.end()){
                    if(j != 0)
                        fprintf(output,"\t\t%i -> %i [label=\"%i\"];\n", i, j, a);
                    //++it;
                }
                if(y[a][i][j] == -1){
                    if(j != 0)
                        fprintf(output,"\t\t%i -> %i [label=\"%i\"];\n", i, j, a);
                }
            }
        }
    }
    
    for(a = 0; a < alphabet_size; ++a){
        for(i = 0; i < dfa_size; ++i){
            for(j = 0; j < sinks_size; ++j){
                if(trueliterals.find(y[a][i][j]) != trueliterals.end()){
                    //++it;
                    fprintf(output,"\t\t %i -> s%i [label=\"%i\"];\n", i, j, a);
                }
            }
        }
    }

    for(j = 0; j < sinks_size; ++j){
        fprintf(output,"\ts%i [shape=box];\n", j);
    }

    /*
    for(i = 0; i < num_states; ++i){
        if(sp[i] == *it){
            //cerr << "sp " << i << endl;
            ++it;
        }
    }
     */
    
    /*
    for(i = 0; i < dfa_size; ++i){
        if(z[i] == *it){
            ++it;
            fprintf(output,"\t%i [shape=doublecircle];\n", i);
        } else if(z[i] == -1){
            fprintf(output,"\t%i [shape=doublecircle];\n", i);
        } else {
            fprintf(output,"\t%i [shape=Mcircle];\n", i);
        }
    }
    */
    
    fprintf(output,"}\n");
    fclose(output);
}

/* output result to aut, for later processing in i.e. ensembles */
void dfasat::print_aut_output(const char* aut_output){
    FILE* output = fopen(aut_output, "w");
    apta* aut = merger->get_aut();
    int v,i,a,j;
    
    fprintf(output,"%i %i\n", dfa_size, alphabet_size);
    fprintf(output,"%i\n", state_number[aut->get_root()->find()]);
    
    std::set<int>::iterator it = trueliterals.begin();
    for(v = 0; v < num_states; ++v)
        for(i = 0; i < dfa_size; ++i)
            if(x[v][i] == *it) ++it;
    
    for(a = 0; a < alphabet_size; ++a){
        for(i = 0; i < dfa_size; ++i){
            for(j = 0; j < dfa_size; ++j) {
                if(y[a][i][j] == *it){
                    fprintf(output,"t %i %i %i\n", i, a, j);
                    ++it;
                }
                if(y[a][i][j] == -1){
                    fprintf(output,"t %i %i %i\n", i, a, j);
                }
            }
        }
    }
    
    for(a = 0; a < alphabet_size; ++a){
        for(i = 0; i < dfa_size; ++i){
            for(j = 0; j < sinks_size; ++j){
                if(sy[a][i][j] == *it){
                    ++it;
                    fprintf(output,"t %i %i %i;\n", i, a, dfa_size+j);
                }
            }
        }
    }
    
    for(i = 0; i < num_states; ++i){
        if(sp[i] == *it){
            //cerr << "sp " << i << endl;
            ++it;
        }
    }
    
    /*
    for(i = 0; i < dfa_size; ++i){
        if(z[i] == *it){
            ++it;
            fprintf(output,"a %i 1\n", i);
        } else if(z[i] == -1){
            fprintf(output,"a %i 1\n", i);
        } else {
            fprintf(output,"a %i 0\n", i);
        }
    }
    */

    for(i = 0; i < sinks_size; ++i){
        fprintf(output,"a %i s%i\n", i+dfa_size, i);
    }
    fclose(output);
}

dfasat::dfasat(state_merger* m, int best_solution){
    merger = m;
    state_set *all_states = merger->get_all_states();
    ag = new apta_graph(all_states);
    //ag->add_conflicts(merger);
    //ag->extract_types(50);

    alphabet_size = inputdata_locator::get()->get_alphabet_size();

    red_states = merger->get_red_states();
    non_red_states = merger->get_candidate_states();
    sink_states = merger->get_sink_states();

    if(best_solution != -1)
        dfa_size = best_solution - 1;
    else
        dfa_size = red_states->size() + OFFSET;

    sinks_size = 0;
    if (USE_SINKS) sinks_size = merger->get_aut()->get_root()->get_data()->num_sink_types();

    if (!MERGE_SINKS_PRESOLVE) non_red_states->insert(sink_states->begin(), sink_states->end());
    num_states = red_states->size() + non_red_states->size();

    if (best_solution != -1) dfa_size = std::min(dfa_size, best_solution);
    new_states = dfa_size - red_states->size();
    new_init = red_states->size();

    num_types = 2;//ag->num_types;

    /* run reduction code IF valid solver was specified */

    /*
    struct stat buffer;
    bool sat_program_exists = (stat(sat_program.c_str(), &buffer) == 0);
    if (sat_program != "" && sat_program_exists) {
    */
    // assign a unique number to every state
    int i = 0;
    for (state_set::iterator it = red_states->begin(); it != red_states->end(); ++it) {
        apta_node *node = *it;
        state_number[node] = i;
        number_state[i] = node;
        state_colour[node] = i;
        i++;
    }
    for (state_set::iterator it = non_red_states->begin(); it != non_red_states->end(); ++it) {
        apta_node *node = *it;
        state_number[node] = i;
        number_state[i] = node;
        i++;
    }

    clause_counter = 0;
    literal_counter = 1;

    std::cerr << "creating literals..." << std::endl;
    create_literals();
}

void dfasat::compute_header() {
    std::cerr << "number of red states: " << red_states->size() << std::endl;
    std::cerr << "number of non_red states: " << non_red_states->size() << std::endl;
    std::cerr << "number of sink states: " << sink_states->size() << std::endl;
    std::cerr << "dfa size: " << dfa_size << std::endl;
    std::cerr << "sink types: " << sinks_size << std::endl;
    std::cerr << "new states: " << new_states << std::endl;
    std::cerr << "new init: " << new_init << std::endl;

    fix_red_values();
    if (USE_SINKS) fix_sink_values();
    erase_red_conflict_colours();
    set_symmetry();

    // renumber literals to account for eliminated ones
    reset_literals(false);

    computing_header = true;

    clause_counter = 0;
    clause_counter += print_colours();
    clause_counter += print_conflicts();
    clause_counter += print_accept();
    clause_counter += print_transitions();
    std::cerr << "total clauses before symmetry: " << clause_counter << std::endl;
    if (SYMMETRY_BREAKING) {
        std::cerr << "Breaking symmetry in SAT" << std::endl;
        clause_counter += print_t_transitions();
        clause_counter += print_p_transitions();
        clause_counter += print_a_transitions();
        clause_counter += print_symmetry();
    }
    if (FORCING) {
        std::cerr << "Forcing in SAT" << std::endl;
        clause_counter += print_forcing_transitions();
    }

    clause_counter += print_paths();
    if (USE_SINKS) {
        clause_counter += print_sink_transitions();
        clause_counter += print_sink_paths();
    }
    std::cerr << "header: p cnf " << literal_counter - 1 << " " << clause_counter << std::endl;
}

void dfasat::translate(FILE* sat_file) {
    computing_header = false;
    sat_stream.clear();

    //fprintf(sat_stream, "p cnf %i %i\n", literal_counter - 1, clause_counter);
    sat_stream << "p cnf " << literal_counter - 1 << " " << clause_counter << "\n";

    print_colours();
    print_conflicts();
    print_accept();
    print_transitions();
    if(SYMMETRY_BREAKING){
        print_symmetry();
        print_t_transitions();
        print_p_transitions();
        print_a_transitions();
    }
    if(FORCING){
        print_forcing_transitions();
    }
    print_paths();
    if(USE_SINKS) {
        print_sink_transitions();
        print_sink_paths();
    }

    fprintf(sat_file, "%s", sat_stream.str().c_str());
    fclose(sat_file);

    std::cerr << "sent problem to SAT solver" << std::endl;
}

void dfasat::perform_sat_merges(state_merger* m) {
    std::map<int,apta_node*> color_node;
    apta* aut = m->get_aut();
    red_state_iterator itr = red_state_iterator(aut->get_root());
    while(*itr != nullptr) {
        apta_node *red = *itr;
        int nr = state_number[red];
        int cr = -1;

        for (int j = 0; j < dfa_size; ++j) {
            if (x[nr][j] == -1) {
                cr = j;
                break;
            }
            if (trueliterals.contains(x[nr][j])) {
                cr = j;
                break;
            }
        }

        if (cr == -1) {
            std::cerr << "error performing merges" << std::endl;
            break;
        }

        if (!color_node.contains(cr)) {
            color_node[cr] = red;
            //std::cerr << "coloring node red " << cr << std::endl;
        }
        ++itr;
    }

    blue_state_iterator it = blue_state_iterator(aut->get_root());
    while(*it != nullptr){
        apta_node *blue = *it;
        int nr = state_number[blue];
        int cr = -1;

        for(int j = 0; j < dfa_size; ++j) {
            if (x[nr][j] == -1) {
                cr = j;
                break;
            }
            if (trueliterals.contains(x[nr][j])) {
                cr = j;
                break;
            }
        }

        if(cr == -1) {
            std::cerr << "error performing merges" << std::endl;
            break;
        }

        if(!color_node.contains(cr)) {
            m->extend(blue);
            color_node[cr] = blue;
            //std::cerr << "coloring node blue " << cr << std::endl;
        } else {
            apta_node* red = color_node[cr];
            m->perform_merge(red, blue);
        }
        it = blue_state_iterator(aut->get_root());
    }
}

void dfasat::read_solution(FILE* sat_file, int best_solution, state_merger* merger) {
    trueliterals = std::set<int>();

    char line[5000];

    bool improved = false;
    bool read_v = false;
    char* broken_val = nullptr;
    while (fgets(line, sizeof line, sat_file) != NULL) {
        //std::cerr << line << std::endl;
        char *pch = strtok(line, " ");
        if (strcmp(pch, "s") == 0) {
            pch = strtok(NULL, " ");
            std::cerr << pch << std::endl;
            if (strcmp(pch, "SATISFIABLE\n") == 0) {
                std::cerr << "new solution, size = " << dfa_size << std::endl;
                if (best_solution == -1 || best_solution > dfa_size) {
                    std::cerr << "new best solution, size = " << dfa_size << std::endl;
                    best_solution = dfa_size;
                    improved = true;
                }
            }
        } else if (read_v || strcmp(pch, "v") == 0) {
                read_v = true;
                //std::cerr << pch << std::endl;
                pch = strtok(NULL, " ");
                if(broken_val != nullptr){
                    std::string combined = std::string(broken_val) + std::string(pch);
                    int val = atoi(pch);
                    if (val > 0) trueliterals.insert(val);
                    pch = strtok(NULL, " ");
                }
                int prev_val = 0;
                while (pch != NULL) {
                    int val = atoi(pch);
                    if (abs(prev_val) > abs(val)){
                        broken_val = pch;
                    } else {
                        if (val > 0) trueliterals.insert(val);
                    }
                    pch = strtok(NULL, " ");
                    prev_val = val;
                }
        }
    }
    perform_sat_merges(merger);
    //print_dot_output("satout.dot");
    fclose(sat_file);

    delete_literals();
}

void start_sat_solver(std::string sat_program){
#ifdef _WIN32
    std::cerr << "DFASAT does not work under Windows OS" << std::endl;
#else
    std::cerr << "starting SAT solver " << sat_program << std::endl;
    char* copy_sat = strdup(sat_program.c_str());
    char* pch = strtok (copy_sat," ");
    std::vector<char*> args;
    while (pch != NULL){
        args.push_back(strdup(pch));
        pch = strtok (NULL," ");
    }
    free(copy_sat);
    free(pch);
    args.push_back((char*)NULL);
    execvp(args[0], &args[0]);
    std::cerr << "finished SAT solver" << std::endl;
    for(int argi = 0; argi < args.size(); ++argi) free(args[argi]);
    int status;
    wait(&status);
    //WIFEXITED(&status);
#endif // WIN32
}


/* the main routine:
 * run greedy state merging runs
 * convert result to satisfiability formula
 * run sat solver
 * translate result to a DFA
 * print result
 * */
void run_dfasat(state_merger* m, std::string sat_program, int best_solution) {
#ifdef _WIN32
    std::cerr << "DFASAT does not work under Windows OS" << std::endl;
#else
    sat_program = SAT_SOLVER;
    if(SAT_SOLVER.compare("") == 0){
        sat_program = "./glucose -model";
    }

    std::cerr << "calling " << sat_program << std::endl;

    dfasat sat_object = dfasat(m, best_solution);
    sat_object.compute_header();

    /* CODE TO RUN SATSOLVER AND CONNECT USING PIPES, ONLY WORKS UNDER LINUX */

        int pipetosat[2];
        int pipefromsat[2];
        if (pipe(pipetosat) < 0 || pipe(pipefromsat) < 0){
            std::cerr << "Unable to create pipe for SAT solver: " << strerror(errno) << std::endl;
            exit(1);
        }

        pid_t pid = fork();
        if (pid == 0) {
            close(pipetosat[1]);
            dup2(pipetosat[0], STDIN_FILENO);
            close(pipetosat[0]);

            close(pipefromsat[0]);
            dup2(pipefromsat[1], STDOUT_FILENO);
            close(pipefromsat[1]);

            start_sat_solver(sat_program);
        }
            else {
            FILE *sat_file = (FILE *) fdopen(pipetosat[1], "w");
            //FILE* sat_file = (FILE*) fopen("test.out", "w");
            if (sat_file == 0) {
                std::cerr << "Cannot open pipe to SAT solver: " << strerror(errno) << std::endl;
                exit(1);
            }

            std::cerr << "sending problem...." << std::endl;

            sat_object.translate(sat_file);

            close(pipetosat[0]);
            close(pipefromsat[1]);

            time_t begin_time = time(nullptr);

            sat_file = (FILE *) fdopen(pipefromsat[0], "r");

            sat_object.read_solution(sat_file, best_solution, m);

            std::cerr << "solving took " << (time(nullptr) - begin_time) << " seconds" << std::endl;
        }
#endif // WIN32
};

