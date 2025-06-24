
#ifndef _ENSEMBLE_H_
#define _ENSEMBLE_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <list>
#include "state_merger.h"
#include "refinement.h"

class Model {
public:
    Model(state_merger* pta, double rd, double weight = 0);

    void assign_weights(std::vector<double>& weights);
    void train();
    std::pair<std::vector<trace*>, double> validate(std::string validation_file, int i, inputdata & idat) const;
    void update_weights(const std::vector<trace*>& traces, std::vector<double> & weights, state_merger* apta_merger);

    int predict(trace *tr) const;
    double evaluateAccuracy(std::string test_file, std::string apta_file, int i);

    void setRandom(double r);
    double getRandom() const;
    void setWeight(double w);
    double getWeight() const;
    json print_json() const;

private:
    state_merger* merger;
    double rd;
    double weight;
};

class Ensemble {
public:
    void addModel(const Model& model);
    void evaluateModelWeights(std::string test_file, std::string apta_file);
    int predict(trace *tr) const;
    double evaluateAccuracy(std::string test_file);

    void tojson();
    void print_json(const std::string& file_name);
    void read_json(std::ifstream & in, inputdata* id, evaluation_function* eval);

private:
    std::vector<Model> models;
    std::string json_output;
};

void bagging(state_merger* merger, std::string output_file, int nr_estimators);

void ensemble_random(inputdata* id, evaluation_function* eval, int nr_estimators, double lb, double ub, std::string output_file);
void ensemble_boosting(inputdata* id, evaluation_function* eval, int nr_estimators, std::string output_file, std::string validation_file, int n_states);

void predict_ensemble(std::ifstream &in, inputdata *id, evaluation_function *eval, std::string valid_file, std::string test_file, std::string apta_file);

#endif /* _ENSEMBLE_H_ */
