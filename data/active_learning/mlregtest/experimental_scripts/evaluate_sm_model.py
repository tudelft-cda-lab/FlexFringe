#!/usr/bin/env python
# coding: utf-8

import os

import pandas as pd
import numpy as np

from sklearn.metrics import balanced_accuracy_score, accuracy_score

import argparse

type_mapping_dict = {
  "A": 0,
  "R": 1,
}

def map_type(x):
  x = x.strip()
  if x in type_mapping_dict:
    return type_mapping_dict[x]
  return -1 

def count_states(model_filename):
  """Countes the number of states that the model has. 

  At the moment: Take the .dot file, and count the number of lines 
  that have 'path' substring in them. Reason: Easy to implement.

  Args:
      model_filename (string): The path to the model.
  """
  res = 0
  for line in open(model_filename, "rt"):
    if "path" in line:
      res += 1
  return res

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("problem_id", type=str, help="The ID of a MLRegTest problem. Example: 04.04.PT.2.1.1")
  parser.add_argument("model_path", type=str, help="Path to where the state machine model resides, or the path to the model itself. In the first case the model filename is derived via [problem_id]_Train.txt.dat.ff.final.json.")
  parser.add_argument("--min_test_length", type=int, default=0, help="The minimum length of test strings. If provided only test on strings larger. No value means all strings.")
  parser.add_argument("--test_data_path", type=str, default=None, help="The path where the test-data resides. If not provided assumed to be in model_path")
  parser.add_argument("--output_dir", type=str, default=None, help="Path of where to store the results. If not provided implicitely make directory 'sm_results'\
                      in this scripts directory and store there.")

  args = parser.parse_args()

  problem = args.problem_id

  model_path = args.model_path if not args.model_path.endswith(".json") else "./" + "/".join(os.path.split(args.model_path)[:-1])

  # the follwing blocks remain fixed
  MODEL_FILENAME = "{}_Train.txt.dat.ff.final.json".format(problem) # probably not needed
  MODEL_FILENAME_DOT = "{}_Train.txt.dat.ff.final.json".format(problem) if not args.model_path.endswith(".result") else os.path.split(args.model_path)[-1][:-7]
  TEST_DATA_FILENAME = "{}_TestSR.txt.dat".format(problem)
  RESULTS_FILENAME = "{}.result".format(MODEL_FILENAME_DOT)
  #RESULTS_FILENAME = "{}_TestSR.txt.dat.ff.result".format(problem)

  #data/active_learning/mlregtest/data/abbadingo/Short_Sequences_Mid/04.04.PT.6.1.9_Train.txt.dat.ff.final.json.result

  TEST_DATA_DIR = args.test_data_path if args.test_data_path else model_path

  predicted = list()
  true = list()
  results_file_path = os.path.join(TEST_DATA_DIR, RESULTS_FILENAME) if not args.model_path.endswith(".result") else args.model_path
  results_df = pd.read_csv(results_file_path, delimiter=";")

  n_unknown = len(results_df.loc[results_df[' trace type'] == " 0"])
  results_df = results_df.loc[results_df[" predicted trace type"] != " 0"]
  
  true_types = results_df[" trace type"].apply(map_type).to_numpy()
  predicted_types = results_df[" predicted trace type"].apply(map_type).to_numpy()

  if args.min_test_length > 0:
    for idx in range(len(results_df)):
      if len(results_df[" score sequence"][idx].strip().strip("[").strip("]").split(",")) > args.min_test_length:
        true_types = true_types[idx:]
        predicted_types = predicted_types[idx:]
        break

  acc = accuracy_score(true_types, predicted_types)
  bacc = balanced_accuracy_score(true_types, predicted_types)
  json_model_path = os.path.join(args.model_path, MODEL_FILENAME_DOT) if not args.model_path.endswith(".result") else args.model_path[:-7]
  n_states = count_states(json_model_path)
  
  OUTPUT_DIR = "sm_results" if not args.output_dir else args.output_dir
  if not os.path.isdir(OUTPUT_DIR):
     os.mkdir(OUTPUT_DIR)
  OUTPUT_NAME = "{}_state_merging.txt".format(problem)
  with open(os.path.join(OUTPUT_DIR, OUTPUT_NAME), "wt") as outf_h:
     outf_h.write("accuracy | {}\nbalanced accuracy | {}\nn_unknow | {}\nn_states | {}".format(acc, bacc, n_unknown, n_states))