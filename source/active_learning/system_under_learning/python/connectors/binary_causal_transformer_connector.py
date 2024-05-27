"""
Connects to the models we trained on the MLRegTest dataset. Assumes that both the dataset metadata as well as the trained
model are in the same directory.

Documentation of CPython-API: https://docs.python.org/3/c-api/index.html

Created by Robert Baumgartner, 27.5.24.
"""

import os
import pickle as pk

import re

import torch
import torch.nn as nn
from torch.nn.functional import softmax

import transformers

import sys
#sys.path.append("/home/robert/Documents/code/Flexfringe/source/active_learning/system_under_learning/python/util")
sys.path.append("../util")
from distillbert_for_language_model import DistilBertForTokenClassification
#from transformers import DistilBertForTokenClassification

DEVICE = "cpu" # we do not do batch processing, so CPU should be good enough

# we need these variables globally, as they represent the status of our program
model = None 

# the following will be set by get_alphabet()
SOS = None
EOS = None
PAD = None 

MAXLEN = None # max length of a sequence. Used for padding and set by get_alphabet() function
ALPHABET_SIZE = None


def make_dict(**kwargs):
    return kwargs

def init_global_variables(path_to_model: str):
  global MAXLEN, ALPHABET_SIZE, SOS, EOS, PAD

  path_split = os.path.split(path_to_model)
  data_dir = os.path.join(*path_split[:-1])

  regex_pattern = "\d\d\.\d\d\.[A-Z]{2,5}\.\d\.\d\.\d"
  problem_id = re.search(regex_pattern, path_to_model).group(0)
  print("Identified problem {}".format(problem_id))

  path_to_dataset = os.path.join(data_dir, "dataset_problem_{}.pk".format(problem_id))
  dataset_dict = pk.load(open(path_to_dataset, "rb"))

  ALPHABET_SIZE = dataset_dict["alphabet_size"]
  SOS = dataset_dict["SOS"]
  EOS = dataset_dict["EOS"]
  PAD = dataset_dict["PAD"]
  MAXLEN = dataset_dict["maxlen"]

def load_nn_model(path_to_model: str):
  """Loads a model and writes it into the global model-variable

  Args:
      path_to_model (str): Full absolute path to the model. In flexfringe, this is
      the aptafile-argument.
  """
  global model
  init_global_variables(path_to_model)

  init_dict = make_dict( # WARNING: must be same as in training file
      # vocab_size=train_dataset.alphabet_size+3,
      vocab_size=ALPHABET_SIZE+5,
      num_labels=ALPHABET_SIZE+5,
      max_position_embeddings=MAXLEN+2,
      sinusoidal_pos_embds=True,
      use_pos_embds=True,
      
      n_layers=4,
      n_heads=12,
      # dim=train_dataset.alphabet_size*2,
      dim = 96,
      hidden_dim = 96,
      # hidden_dim=train_dataset.alphabet_size*2,
      activation="gelu",
      dropout=0.1,
      attention_dropout=0.1,
      seq_classif_dropout=0.2,
      PAD_TOKEN_id=PAD
  )

  model = DistilBertForTokenClassification(transformers.DistilBertConfig(**init_dict))
  print("Load state dict")
  model.load_state_dict(torch.load(path_to_model, map_location=torch.device(DEVICE)))
  print("Params")
  for param in model.parameters():
      param.requires_grad = False

def make_tensor_causal_masks(words:torch.Tensor):
    masks = (words != PAD) # 1 : pass, 0 : blocked
    b,l = masks.size()
    x = torch.einsum("bi,bj->bij",masks,masks)
    # Mask with lower triangle including diagonal -> causal mask
    x *= torch.ones(l,l, dtype=torch.bool, device=x.device).tril() 
    x += torch.eye(l,dtype=torch.bool, device=x.device)
    return x.type(torch.int8)

def do_query(seq: list):
  """This is the main function, performed on a sequence.
  Returns what you want it to return, make sure it makes 
  sense in the employed SUL class. 

  Args:
      seq (list): List of ints.
  """
  padding = [PAD] * (MAXLEN - len(seq) - 1)
  padded_seq = seq + [EOS] + padding
  last_token = len(seq) - 1

  query_string = torch.reshape(torch.tensor(padded_seq), (1, -1))
  mask = make_tensor_causal_masks(query_string)
  with torch.no_grad():
    output = model(input_ids=query_string, attention_mask=mask, return_dict=True, output_attentions=False)
  logits = torch.squeeze(output.logits)

  last_pos = torch.where(query_string == EOS)[1]#[0]
  print(query_string, EOS, last_pos)
  preds = torch.argmax(logits[last_pos])

  if(logits.size(-1) > PAD + 1):
      preds = preds - PAD - 1
  
  print(preds)
  return preds
  #if torch.any(preds > 0):
  #   return 1
  #return 0


def get_alphabet(path_to_model: str):
  """Returns the alphabet, so we can set the internal alphabet of Flexfringe accordingly.
  Alphabet must a list of int objects, representing the possible inputs of the network. 
  Flexfringe will take care of the internals.

  Args:
      path_to_model (str): Full absolute path to the model. In flexfringe, this is
      the aptafile-argument (used to infer the path to the alphabet. Must be in same dir.).

  Returns:
      alphabet: list(int): The possible inputs the network can take.
  """
  path_split = os.path.split(path_to_model)
  data_dir = os.path.join(*path_split[:-1])
  
  regex_pattern = "\d\d\.\d\d\.[A-Z]{2,5}\.\d\.\d\.\d"
  problem_id = re.search(regex_pattern, path_to_model).group(0)

  path_to_dataset = os.path.join(data_dir, "dataset_problem_{}.pk".format(problem_id))
  dataset_dict = pk.load(open(path_to_dataset, "rb"))
  symbol_dict = dataset_dict["symbol_dict"]

  alphabet = [int(v) for k, v in symbol_dict.items() if not k=="<SOS>" and not k=="<EOS>"]
  assert(alphabet is not None)
  print("The alphabet: ", alphabet)
  return alphabet


if __name__ == "__main__":
  model_path = "/home/robert/Documents/code/Flexfringe/data/active_learning/mlregtest/trained_models/distilbert_problem_16.16.SL.2.1.9_1.pk.simul"

  load_nn_model(model_path)
  seq = [4, 0]
  res = do_query(seq)
  print(res)

  raise Exception("This is not a standalone script.")