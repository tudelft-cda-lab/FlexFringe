/**
 * @file css.h
 * @author Robert Baumgartner, Raffail Skoulos
 * @brief This file implements count-min-sketch based state merging.
 * @version 0.1
 * @date 2020-05-29
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef __CSS__
#define __CSS__

#include "alergia.h"
#include "countminsketch.h"

#include <string>

/* The data contained in every node of the prefix tree or DFA */
class css_data : public alergia_data {
protected:

    class minhash_func {
    private:
        const std::vector<int> input_alphabet;
        std::map<int, int> permutation;

    public:
        minhash_func(const std::vector<int> alphabet) : input_alphabet(alphabet){
            for(const auto symbol: alphabet){
                permutation[symbol] = permutation.size(); 
            }
        }

        const int get_mapping(const std::set<int> shingle_set) const {
            for(const auto symbol: input_alphabet){
                if(shingle_set.count(symbol) > 0){
                    return permutation.at(symbol);
                }
            }
            std::cerr << "Unknown symbol found in shingle. This should not have happened. Is the estimated alphabet too small?" << std::endl;
            throw new std::runtime_error("ERR");
        }
    };

    REGISTER_DEC_DATATYPE(css_data);

    inline static std::map<int, double> symbol_to_mapping; // only needed for conditional probabilities
    inline static std::vector<minhash_func> minhash_functions;
    inline static std::vector< std::set<int> > seen_symbols;

    std::vector<CountMinSketch> sketches;

    const std::vector<int> get_symbols_as_list(tail* t) const;
    const std::vector<int> minhash_set(const std::set<int> shingle_set, const int n_gram_size) const;
    const std::vector<int> encode_sets(const std::vector< std::set<int> > shingle_sets) const;

public:

    css_data();

    inline const std::vector< CountMinSketch >& get_sketches() const {return this->sketches; }

    virtual void update(evaluation_data *right) override;
    virtual void undo(evaluation_data *right) override;

    void add_tail(tail* t) override;

    //************************************* Sinks **********************************************

    inline bool is_low_count_sink() const noexcept {
        return this->node->get_size() < SINK_COUNT;
    }

    //inline bool is_stream_sink(apta_node* node) const noexcept {
    //    return this->node->get_size() < STREAM_COUNT;
    //}

    inline int sink_type(apta_node* node) const noexcept {
        if(!USE_SINKS) return -1;

        if (is_low_count_sink()) return 0;
        //if(is_stream_sink(node)) return 2;
        return -1;
    };

    inline bool sink_consistent(int type) const noexcept {
        if(!USE_SINKS) return true;

        if(type == 0) return is_low_count_sink();
        return true;
    };

    inline int num_sink_types() const noexcept{
        if(!USE_SINKS) return 0;
        return 2;
    };

    inline const std::set<int>& get_seen_symbols(const int i) const {
        return seen_symbols.at(i);
    }

    void print_state_label(std::iostream& output);
};

class css : public alergia {

protected:
    REGISTER_DEC_TYPE(css);
    double scoresum = 0;

public:
    virtual double compute_score(state_merger *, apta_node *left, apta_node *right);
    virtual void reset(state_merger *merger);
    virtual bool consistent(state_merger *merger, apta_node *left, apta_node *right);
};

#endif
