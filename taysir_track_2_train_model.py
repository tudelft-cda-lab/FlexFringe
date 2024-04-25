#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
import shutil
import subprocess

result_dir = "taysir_results_track_2"
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

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

def get_start_and_end_symbol(i, path_to_train_sets):
    train_set_path = os.path.join(path_to_train_sets, "2.{}.taysir.valid.words".format(i))
    with open(train_set_path, "rt") as f:
        _ = f.readline()
        line = f.readline()
        line = line.strip().split()
        start_symbol = line[1]
        end_symbol = line[-1]
    return start_symbol, end_symbol    


# ## Train the models

# In[ ]:

if __name__ == "__main__":
    model_nr = int(sys.argv[1])
    print("Model number: {}".format(model_nr))
    tmp_script_path = transformer_script_path if model_nr==10 else nn_script_path
        
    model_name = "2.{}.taysir.model".format(model_nr)
    model_path = os.path.join(path_to_models, model_name)
        
    start_symbol, end_symbol = get_start_and_end_symbol(model_nr, path_to_train_sets)
        
    #print("Starting training model nr. {}. Start-symbol={}, end-symbol={}".format(model_nr, start_symbol, end_symbol))
    #command = ["./flexfringe", "--ini={}".format(ini_path), "--start_symbol={}".format(start_symbol), \
    #        "--end_symbol={}".format(end_symbol), "--aptafile={}".format(model_path), tmp_script_path]
    #p = subprocess.run(command, stdout=subprocess.PIPE,
    #                            stderr=subprocess.PIPE, universal_newlines=True)
    #for outstr in p.stdout:
    #    sys.stdout.write(outstr)
    #for outstr in p.stderr:
    #    sys.stderr.write(outstr)
    #print("Finished training model nr. {}. Starting prediction.".format(model_nr))
    #    
    test_set_name = "2.{}.taysir.test.combined".format(model_nr)
    test_set_path = os.path.join(path_to_test_sets, test_set_name)
    command = ["./flexfringe", "--ini={}".format(pred_ini_path), \
            "--aptafile={}{}".format(tmp_script_path, json_ending), test_set_path]
    p = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, universal_newlines=True)
    for outstr in p.stderr:
        sys.stderr.write(outstr)
    print("Finished prediction. Move the files to the folder now.")
            
    shutil.move(tmp_script_path + json_ending, os.path.join(result_dir, "taysir_model_{}.json".format(model_nr)))
    shutil.move(tmp_script_path + dot_ending, os.path.join(result_dir, "taysir_model_{}.dot".format(model_nr)))
    shutil.move(tmp_script_path + result_ending, os.path.join(result_dir, "taysir_model_{}.result".format(model_nr)))
        
    #command = ["dot", "-Tpdf", os.path.join(result_dir, "taysir_model_{}.dot".format(i)), ">>", \
    #           os.path.join(result_dir, "taysir_model_{}.pdf".format(i))]
    #p = subprocess.run(command, stdout=subprocess.PIPE,
    #                            stderr=subprocess.PIPE, universal_newlines=True)
    #for outstr in p.stderr:
    #    sys.stderr.write(outstr)
    print("Done with model nr. {}".format(model_nr))

    exit(0)
