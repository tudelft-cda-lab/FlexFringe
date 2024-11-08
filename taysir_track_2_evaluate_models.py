import json
import pandas as pd

import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score

import os, sys
import shutil
import subprocess

RANGE = [1, 10]

result_dir = "taysir_results_track_2"
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

def get_score(s: str):
  res = 1
  for value_str in s.strip( "'[] " ).split(","):
      res *= float(value_str)
  return res

# config
nn_script_path = "source/active_learning/system_under_learning/python/nn_connector_taysir_track_1.py"
transformer_script_path = "source/active_learning/system_under_learning/python/nn_connector_taysir_track_2_transformer.py"
ini_path = "ini/probabilistic_lsharp.ini"
pred_ini_path = "ini/predict_probabilistic_lsharp.ini"

# dataset and models
path_to_models = "data/active_learning/test_nn_queries/taysir_competition/track2/models"
path_to_test_sets = "data/active_learning/test_nn_queries/taysir_competition/track2/test_sets"
path_to_train_sets = "data/active_learning/test_nn_queries/taysir_competition/track2/datasets"

json_ending = ".ff.final.json"
dot_ending = ".ff.final.dot"
result_ending = ".ff.final.json.result"

summary_fh = open(os.path.join(result_dir, "results_summary.txt"), "wt")
summary_fh.write("Model nr. {}, Mean, Sum, Max, MSE\n")
for i in range(RANGE[0], RANGE[-1]+1):
    print("Predicting model {}".format(i))
    label_file = os.path.join(path_to_test_sets, "2.{}.taysir.test.labels".format(i))
    labels = list()
    for j, label in enumerate(open(label_file, "rt")):
        if j==0:
            continue
        label = label.strip().strip("\n").strip()
        labels.append(float(label))
        
    result_file = os.path.join(result_dir, "taysir_model_{}.result".format(i))
    res_df = pd.read_csv(result_file, delimiter=";")
        
    res_df["total_scores"] = res_df[" score sequence"].map(get_score)
    
    diffs = np.abs(np.array(res_df["total_scores"]).reshape(-1) - np.array(labels).reshape(-1))
    mse = np.sqrt(np.sum(diffs * diffs)) / diffs.shape[0]
    print(mse)
    outstr = "{}, {}, {}, {}, {}\n".format(i, np.mean(diffs), np.sum(diffs), np.max(diffs), mse)
    summary_fh.write(outstr)
    summary_fh.flush()
summary_fh.close()