#include <sstream>
#include <fstream>
#include <cstdlib>

#include "refinement.h"
#include "greedy.h"
#include "parameters.h"
#include "ensemble.h"
#include "predict.h"
#include "csv.hpp"
#include "input/parsers/abbadingoparser.h"
#include "input/inputdatalocator.h"

/** todo: work in progress */

refinement_list* greedy(state_merger* merger){
    std::cerr << "starting greedy merging" << std::endl;
    merger->get_eval()->initialize_after_adding_traces(merger);

    auto* all_refs = new refinement_list();

    refinement* best_ref = merger->get_best_refinement();
    while( best_ref != nullptr ){
        std::cout << " ";
        best_ref->print_short();
        std::cout << " ";
        std::cout.flush();

        best_ref->doref(merger);
        all_refs->push_back(best_ref);
        best_ref = merger->get_best_refinement();
    }
    std::cout << "no more possible merges" << std::endl;
    return all_refs;
};

void bagging(state_merger* merger, std::string output_file, int nr_estimators){
    std::cerr << "starting bagging" << std::endl;
    for(int i = 0; i < nr_estimators; ++i){
        refinement_list* all_refs = greedy(merger);

        for(refinement_list::reverse_iterator it = all_refs->rbegin(); it != all_refs->rend(); ++it){
            (*it)->undo(merger);
        }
        for(refinement_list::iterator it = all_refs->begin(); it != all_refs->end(); ++it){
            (*it)->erase();
        }
        delete all_refs;
    }
    std::cerr << "ended bagging" << std::endl;
};

Model::Model(state_merger* pta, double rd, double w)
        : merger(pta), rd(rd), weight(w) {
    if (rd > 0.0) {
        RANDOMIZE_SCORES = rd;
    } else {
        RANDOMIZE_SCORES = 0;
    }
}

void Model::assign_weights(std::vector<double>& weights) {
    int i = 0;
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    double sum2 = 0;
    for(merged_APTA_iterator Ait = merged_APTA_iterator(merger->get_aut()->get_root()); *Ait != nullptr; ++Ait) {
        weights[i] /= sum;
        (*Ait)->set_weight(weights[i]);
        //std::cout << (*Ait)->get_weight() << " ";
        sum2 += (*Ait)->get_weight();
        i++;
    }
    std::cout << '\n' << "weight: " << sum2 << '\n';
}

void Model::train() {
    std::cout << "size before training: " << merger->get_final_apta_size() << '\n';
    refinement_list * all_refs = greedy(merger);
    std::cout << "size after training: " << merger->get_final_apta_size() << '\n';

    for(refinement_list::iterator it = all_refs->begin(); it != all_refs->end(); ++it){
        (*it)->erase();
    }
    delete all_refs;
}

std::pair<std::vector<trace*>, double> Model::validate(std::string validation_file, int i, inputdata & idat) const {
    int rownr = 0, correct = 0, total = 0;
    std::vector<trace*> misclassified;
    // We stream the to predict traces into inputdata one by one to save memory
    // Set up the parser for the input stream
    std::cout << "Using validation file: " << validation_file << '\n';
    std::ifstream input_stream(validation_file);
    std::unique_ptr<parser> parser;
    parser = std::make_unique<abbadingoparser>(input_stream);

    std::unique_ptr<reader_strategy> strategy;
    if (SLIDING_WINDOW) {
        strategy = std::make_unique<slidingwindow>(SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STRIDE, SLIDING_WINDOW_TYPE);
    } else {
        strategy = std::make_unique<in_order>();
    }

    std::ostringstream res_stream;
    res_stream << validation_file << ".model" << std::to_string(i) << ".result";
    std::ofstream output(res_stream.str().c_str());
    output << "row nr; abbadingo trace; trace type; predicted trace type" << '\n';
    //inputdata idat = inputdata::with_alphabet_from(*inputdata_locator::get());
    std::optional<trace*> trace_maybe = idat.read_trace(*parser, *strategy);

    while (trace_maybe) {
        auto trace = *trace_maybe;
        int pred = predict(trace);
        output << rownr << "; " << "\"" << trace->to_string() << "\"" << "; " << trace->type << "; " << pred << '\n';
        rownr++;
        if (pred == trace->type){
            correct++;
            trace->erase();
        }
        else
            misclassified.push_back(trace);
        total++;
        // TODO: Deleting the traces should probably also invalidate the trace pointers in inputdata,
        //  but since we have a separate inputdata local to this function it is sort of ok here?
        //trace->erase();
        trace_maybe = idat.read_trace(*parser, *strategy);
    }

    return std::make_pair(misclassified, total == 0 ? 0.0 : static_cast<double>(correct) / total);
}

void Model::update_weights(const std::vector<trace *>& traces, std::vector<double> &weights, state_merger* apta_merger) {
    int cnt = 0;
    for (const auto & tr : traces) {
        int i = 0;
        auto state = apta_merger->get_state_from_trace(tr);
        for(merged_APTA_iterator Ait = merged_APTA_iterator(apta_merger->get_aut()->get_root()); *Ait != nullptr; ++Ait) {
            if ((*Ait) == state) {
                weights[i] *= 5;
                cnt++;
                //std::cout << (*Ait)->get_weight() << " ";
                break;
            }
            i++;
        }
        tr->erase();
    }
    std::cout << "cnt: " << cnt << '\n';
}

double Model::evaluateAccuracy(std::string test_file, std::string apta_file, int i) {
    // We stream the to predict traces into inputdata one by one to save memory
    // Set up the parser for the input stream
    std::ifstream input_stream(test_file);
    std::unique_ptr<parser> parser;
    parser = std::make_unique<abbadingoparser>(input_stream);

    std::unique_ptr<reader_strategy> strategy;
    if (SLIDING_WINDOW) {
        strategy = std::make_unique<slidingwindow>(SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STRIDE, SLIDING_WINDOW_TYPE);
    } else {
        strategy = std::make_unique<in_order>();
    }

    std::ostringstream res_stream;
    res_stream << apta_file << std::to_string(i) << ".result";
    std::ofstream output(res_stream.str().c_str());
    predict_streaming(merger, *parser, *strategy, output);

    std::string output_filename = apta_file + std::to_string(i) + ".result";
    std::ifstream input(output_filename);
    if (!input.is_open()) {
        std::cerr << "Failed to open prediction output file: " << output_filename << std::endl;
        return 0.0;
    }

    std::string line;
    std::getline(input, line); //header

    int correct = 0;
    int total = 0;

    while (std::getline(input, line)) {
        std::stringstream ss(line);
        std::string token;
        int col = 0;
        std::string trace_type, predicted_type;

        while (std::getline(ss, token, ';')) {
            if (col == 7) trace_type = token;
            else if (col == 9) predicted_type = token;
            ++col;
        }

        if (!trace_type.empty() && !predicted_type.empty() && trace_type == predicted_type) {
            ++correct;
        }
        ++total;
    }

    return total == 0 ? 0.0 : static_cast<double>(correct) / total;
}

int Model::predict(trace *tr) const {
    return predict_trace_type(merger, tr);
}

json Model::print_json() const {
    json output;
    output["random"] = rd;
    output["accuracy"] = weight;
    merger->tojson();
    json dfa_json = json::parse(merger->json_output);
    output["dfa"] = dfa_json;
    return output;
}

void Model::setRandom(double r) { rd = r; }
double Model::getRandom() const { return rd; }
void Model::setWeight(double w) { weight = w; }
double Model::getWeight() const { return weight; }

void Ensemble::addModel(const Model& model) {
    models.push_back(model);
}

void Ensemble::evaluateModelWeights(std::string test_file, std::string apta_file) {
    for (int i = 0; i < models.size(); i++) {
        double acc = models[i].evaluateAccuracy(test_file, apta_file, i);
        models[i].setWeight(acc);
        std::cout << "model " << i << " acc " << models[i].getWeight() << '\n';
    }
}

int Ensemble::predict(trace *tr) const {
    double vote = 0.0;

    for (const auto& model : models) {
        int prediction = model.predict(tr) == 1 ? 1 : -1;
        vote += prediction * model.getWeight();
    }

    return vote > 0;
}

double Ensemble::evaluateAccuracy(std::string test_file) {
    int rownr = 0, correct = 0, total = 0;
    // We stream the to predict traces into inputdata one by one to save memory
    // Set up the parser for the input stream
    std::ifstream input_stream(test_file);
    std::unique_ptr<parser> parser;
    parser = std::make_unique<abbadingoparser>(input_stream);

    std::unique_ptr<reader_strategy> strategy;
    if (SLIDING_WINDOW) {
        strategy = std::make_unique<slidingwindow>(SLIDING_WINDOW_SIZE, SLIDING_WINDOW_STRIDE, SLIDING_WINDOW_TYPE);
    } else {
        strategy = std::make_unique<in_order>();
    }

    std::ostringstream res_stream;
    res_stream << "ensemble" << ".result";
    std::ofstream output(res_stream.str().c_str());
    output << "row nr; abbadingo trace; trace type; predicted trace type" << '\n';
    inputdata idat = inputdata::with_alphabet_from(*inputdata_locator::get());
    std::optional<trace*> trace_maybe = idat.read_trace(*parser, *strategy);

    while (trace_maybe) {
        auto trace = *trace_maybe;
        int pred = predict(trace);
        output << rownr << "; " << "\"" << trace->to_string() << "\"" << "; " << trace->type << "; " << pred << '\n';
        rownr++;
        if (pred == trace->type)
            correct++;
        total++;
        // TODO: Deleting the traces should probably also invalidate the trace pointers in inputdata,
        //  but since we have a separate inputdata local to this function it is sort of ok here?
        trace->erase();
        trace_maybe = idat.read_trace(*parser, *strategy);
    }

    return total == 0 ? 0.0 : static_cast<double>(correct) / total;
}

void Ensemble::tojson() {
    json ensemble_json = json::array();
    for (const auto& model : models) {
        ensemble_json.push_back(model.print_json());
    }
    json_output = ensemble_json.dump(2);
}

void Ensemble::print_json(const std::string& file_name) {
    tojson();
    std::ofstream output1(file_name.c_str());
    if (output1.fail()) {
        throw std::ofstream::failure("Unable to open file for writing: " + file_name);
    }
    output1 << json_output;
    output1.close();
}

void Ensemble::read_json(std::ifstream &in, inputdata* id, evaluation_function* eval) {
    json ensemble_json = json::parse(in);
    for (const auto& model_json : ensemble_json) {
        double rd = model_json["random"];
        double acc = model_json["accuracy"];

        apta* the_apta = new apta();
        std::stringstream apta_stream;
        apta_stream << model_json["dfa"].dump();

        the_apta->read_json(apta_stream);

        auto* merger = new state_merger(id, eval, the_apta);
        the_apta->set_context(merger);
        eval->set_context(merger);

        Model model(merger, rd, acc);
        this->addModel(model);
    }
}

void ensemble_random(inputdata* id, evaluation_function* eval, int nr_estimators, double lb, double ub, std::string output_file) {
    std::cerr << "starting ensemble random" << std::endl;
    Ensemble ensemble;
    for(int i = 0; i < nr_estimators; ++i) {
        double rd = lb + random_double() * (ub - lb); // rng seeded by default

        apta* the_apta = new apta();
        auto* pta = new state_merger(id, eval, the_apta);
        the_apta->set_context(pta);
        eval->set_context(pta);

        eval->initialize_before_adding_traces();
        id->add_traces_to_apta(the_apta);
        eval->initialize_after_adding_traces(pta);

        Model model(pta, rd);
        model.train();
        ensemble.addModel(model);
    }
    ensemble.print_json(output_file);
    std::cerr << "ended ensembling random" << std::endl;
}

void ensemble_boosting(inputdata* id, evaluation_function* eval, int nr_estimators, std::string output_file, std::string validation_file, int n_states) {
    std::cerr << "starting ensemble boosting" << std::endl;
    Ensemble ensemble;
    std::vector<double> weights(n_states, 1.0);

    for(int i = 0; i < nr_estimators; ++i) {
        apta* the_apta = new apta();
        auto* weighted_pta = new state_merger(id, eval, the_apta);
        the_apta->set_context(weighted_pta);
        eval->set_context(weighted_pta);

        eval->initialize_before_adding_traces();
        id->add_traces_to_apta(the_apta);
        eval->initialize_after_adding_traces(weighted_pta);

        Model model(weighted_pta, -1);
        model.assign_weights(weights);
        model.train();

        apta* the_apta2 = new apta();
        auto* apta_merger = new state_merger(id, eval, the_apta2);
        the_apta2->set_context(apta_merger);
        eval->set_context(apta_merger);

        eval->initialize_before_adding_traces();
        id->add_traces_to_apta(the_apta2);
        eval->initialize_after_adding_traces(apta_merger);

        auto mis_valid_traces = model.validate(validation_file, i + 1, *id);
        model.setWeight(mis_valid_traces.second);
        model.update_weights(mis_valid_traces.first, weights, apta_merger);

        ensemble.addModel(model);
    }
    ensemble.print_json(output_file + ".final_boosting.json");
    std::cerr << "ended ensembling boosting" << std::endl;
}

void predict_ensemble(std::ifstream & in, inputdata* id, evaluation_function* eval, std::string valid_file, std::string test_file, std::string apta_file) {
    Ensemble ensemble;
    ensemble.read_json(in, id, eval);
    if (!BOOSTING)
        ensemble.evaluateModelWeights(valid_file, apta_file);
    std::cout << "Ensemble accuracy: " << ensemble.evaluateAccuracy(test_file) << '\n';
}