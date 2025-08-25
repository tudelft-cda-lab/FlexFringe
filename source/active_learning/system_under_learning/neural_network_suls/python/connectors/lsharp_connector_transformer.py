"""
Like the binary state causal transformer connector, but also returns confidence of transformer as well.

Documentation of CPython-API: https://docs.python.org/3/c-api/index.html

Created by Robert Baumgartner, 10.9.24.
"""

import os
import pickle as pk

import re

import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import softmax

import transformers

import sys
sys.path.append("../util")
from distillbert_for_language_model import DistilBertForTokenClassification
DEVICE = "cpu" # we do not do batch processing, so CPU should be good enough

# we need these variables globally, as they represent the status of our program
model = None 

# the following will be set by get_alphabet()
SOS = None
EOS = None
PAD = None 

MAXLEN = None # max length of a sequence. Used for padding and set by get_alphabet() function
ALPHABET_SIZE = None

ALPH_MAPPING = dict()
INT_TO_TYPE_DICT = dict()


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
  print("Model successfully loaded")

def make_tensor_causal_masks(words:torch.Tensor):
    masks = (words != PAD) # 1 : pass, 0 : blocked
    b,l = masks.size()
    x = torch.einsum("bi,bj->bij",masks,masks)
    # Mask with lower triangle including diagonal -> causal mask
    x *= torch.ones(l,l, dtype=torch.bool, device=x.device).tril() 
    x += torch.eye(l,dtype=torch.bool, device=x.device)
    return x.type(torch.int8)

def do_query(input_seq: list):
  """This is the main function, performed on a sequence.
  Returns what you want it to return, make sure it makes 
  sense in the employed SUL class. 

  Must return a dictionary. For efficiency the following convention holds for the 
  keys: 1 = prediction, 2 = attention

  Args:
      seq (list): List of ints.
  """
  seq = [SOS] +  [ALPH_MAPPING[s] for s in input_seq] + [EOS]
  
  padding = [PAD] * (MAXLEN - len(seq) - 1)
  padded_seq = seq + padding
  last_token_idx = len(seq) - 1

  query_string = torch.reshape(torch.tensor(padded_seq), (1, -1))
  mask = make_tensor_causal_masks(query_string)

  with torch.no_grad():
    model.eval()
    output = model(input_ids=query_string, attention_mask=mask, return_dict=True, output_attentions=False, output_hidden_states=True)
  logits = torch.squeeze(output.logits)

  preds = torch.argmax(logits[last_token_idx])
  if(logits.size(-1) > PAD + 1):
      preds = preds - PAD - 1
  if preds < 0 or preds > 1:
     print("Erroneous prediction: ", preds) # what now?

  if preds.item() in INT_TO_TYPE_DICT: 
    pred_type = INT_TO_TYPE_DICT[preds.item()]
  else:
    print("Transformer did output an invalid type. Changed to unknown type")
    pred_type = "<UNK>"

  res = [pred_type]
  return res


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
  global ALPH_MAPPING, INT_TO_TYPE_DICT

  path_split = os.path.split(path_to_model)
  data_dir = os.path.join(*path_split[:-1])
  
  regex_pattern = "\d\d\.\d\d\.[A-Z]{2,5}\.\d\.\d\.\d"
  problem_id = re.search(regex_pattern, path_to_model).group(0)

  path_to_dataset = os.path.join(data_dir, "dataset_problem_{}.pk".format(problem_id))
  dataset_dict = pk.load(open(path_to_dataset, "rb"))
  symbol_dict = dataset_dict["symbol_dict"]

  print(dataset_dict)

  alphabet = [k for k in symbol_dict.keys() if not k=="<SOS>" and not k=="<EOS>"]
  ALPH_MAPPING = symbol_dict
  INT_TO_TYPE_DICT = {v: k for k, v in dataset_dict["label_dict"].items()}
  
  assert(alphabet is not None)
  print("The alphabet: ", alphabet)
  return alphabet

def get_types():
  """
  Returns the types the transformer can possibly return.
  """
  if not INT_TO_TYPE_DICT or len(INT_TO_TYPE_DICT)==0:
     raise Exception("INT_TO_TYPE_DICT in python script has not been initialized")

  return list(INT_TO_TYPE_DICT.values())
    

if __name__ == "__main__":
  raise Exception("This is not a standalone script.")